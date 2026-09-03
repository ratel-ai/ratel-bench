// The mcpatlas campaign driver.
//
// Bypasses runner.ts and the agents/ registry, same as agent/src/sragents-select.ts
// — this mode spawns live processes (Claude Code, the MCP-Atlas sandbox) and
// mutates live external state, which no in-process `LanguageModel` + `ToolSpec[]`
// harness models. Structure mirrors that driver: a hand-rolled worker pool, a
// version-agnostic cache index read from a prior JSONL, streaming appends rather
// than buffering the whole campaign in memory, and a trivial CLI arg reader.
//
// agent-eval/ imports nothing from agent/src/ (confirmed: every existing
// mcpatlas-*.ts file only imports other mcpatlas-*.ts files). The small
// io/paths helpers below duplicate agent/src/io.ts and agent/src/paths.ts
// rather than reaching across the package boundary — consistent with how
// mcpatlas-stats.ts already duplicates agent/'s stats functions instead of
// importing them.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { LanguageModel } from "ai";
import type { AtlasTool } from "./atlas-mcp-shim.js";
import {
  type ClaudeResult,
  DISALLOWED_TOOLS,
  effectiveCalls,
  parseClaudeResult,
  type RunClaudeOpts,
  type RunClaudeOutcome,
  readTranscript,
  runClaude as realRunClaude,
  transcriptPath,
} from "./mcpatlas-agent.js";
import {
  assembleCell,
  buildRetrievalRows,
  buildSearchEventRows,
  buildToolCallRows,
  type CellContext,
  type ParsedTranscript,
  parseUses,
} from "./mcpatlas-build.js";
import {
  buildCodexArgs,
  buildCodexConfigToml,
  CODEX_PRICING,
  CODEX_TOOL_TIMEOUT_SEC,
  codexInvokeSpans,
  codexLockdown,
  codexResultFromEvents,
  codexRolloutPath,
  countCodexCompactions,
  parseCodexEvents,
  type RunCodexOpts,
  runCodex as realRunCodex,
} from "./mcpatlas-codex.js";
import {
  invokeSpans,
  parseTelemetry,
  perToolTokenMap,
  schemaTokenEstimate,
} from "./mcpatlas-gateway.js";
import { isImmutableRevision } from "./mcpatlas-ingest.js";
import {
  DEFAULT_PARTIAL_THRESHOLD,
  DEFAULT_PASS_THRESHOLD,
  SYSTEM as JUDGE_SYSTEM,
  judgeClaims,
} from "./mcpatlas-judge.js";
import {
  buildPrompt,
  PROMPT_HASH,
  PROMPT_ID,
  SYSTEM_PROMPT_ADDENDUM,
  SYSTEM_PROMPT_ADDENDUM_HASH,
} from "./mcpatlas-prompt.js";
import {
  buildNativeMcpConfig,
  buildRatelMcpConfig,
  buildRatelServeConfig,
  GATEWAY_TOOLS,
  missingEnv,
  normalizeToolId,
  type ShimSpec,
  selectCatalogTools,
  subsetManifest,
} from "./mcpatlas-servers.js";
import type {
  AgentHarness,
  CodexLockdown,
  CodexPricing,
  McpAtlasArm,
  McpAtlasCatalogManifest,
  McpAtlasCell,
  McpAtlasRetrievalRow,
  McpAtlasRunConfig,
  McpAtlasScope,
  McpAtlasSearchEventRow,
  McpAtlasTask,
  McpAtlasToolCallRow,
} from "./mcpatlas-types.js";

// ─────────────────────────────────────────────────────────────────────────────
// Local io/paths — self-contained, see header
// ─────────────────────────────────────────────────────────────────────────────

function findRepoRoot(start: string): string {
  let cur = start;
  for (let depth = 0; depth < 16; depth++) {
    if (existsSync(resolve(cur, "pnpm-workspace.yaml"))) return cur;
    const parent = dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  throw new Error(`could not find repo root walking up from ${start}`);
}

export const REPO_ROOT = findRepoRoot(dirname(fileURLToPath(import.meta.url)));

export function resolveRepoPath(p: string): string {
  return isAbsolute(p) ? p : resolve(REPO_ROOT, p);
}

export function readJsonl<T>(path: string): T[] {
  if (!existsSync(path)) return [];
  const lines = readFileSync(path, "utf8").split("\n");
  const out: T[] = [];
  for (const l of lines) {
    if (!l.trim()) continue;
    out.push(JSON.parse(l) as T);
  }
  return out;
}

export function appendJsonl<T>(path: string, row: T): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(row)}\n`, { flag: "a" });
}

export function truncateJsonl(path: string): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, "");
}

/** Deterministic across key-order permutation — sorts object keys recursively,
 *  leaves array order alone. Used only for config_hash, so the hash doesn't
 *  drift when a field is reordered without meaning to change anything. */
export function stableStringify(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(",")}]`;
  if (v && typeof v === "object") {
    const keys = Object.keys(v as Record<string, unknown>).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify((v as Record<string, unknown>)[k])}`).join(",")}}`;
  }
  return JSON.stringify(v);
}

// ─────────────────────────────────────────────────────────────────────────────
// Config freeze
// ─────────────────────────────────────────────────────────────────────────────

export type FrozenConfigCore = Omit<McpAtlasRunConfig, "config_hash" | "run_id" | "generated_at">;

export interface BuildRunConfigInput {
  scope: McpAtlasScope;
  manifest: McpAtlasCatalogManifest;
  ratelVersionLabel: string;
  ratelLocalVersion: string;
  ratelSdkVersion: string | null;
  claudeCodeVersion: string;
  benchGitSha: string;
  agentModel: string;
  maxTurns: number;
  perCellTimeoutMs: number;
  permissionMode: string;
  /** "" (screen-only) or the model id used for LLM-assisted claim judging. */
  judgeModel: string;
  retrieverMethod: "bm25" | "semantic" | "hybrid";
  topKTools: number;
  topKSkills: number;
  arms: McpAtlasArm[];
  evalKs: number[];
  catalogTools: number;
  runsPerTask: number;
  seed: number;
  concurrency: number;
  datasetRevision: string;
  taskListHash: string;
  taskIds: string[];
  sandboxUrl: string;
  atlasImageDigests: Record<string, string>;
  dollarCapGlobal: number | null;
  declaredLimitations: string[];
  /** Optional with a claude-code default so every pre-harness caller builds
   *  the exact same object it always did — the config-hash golden test pins
   *  this. */
  harness?: AgentHarness;
  codexVersion?: string | null;
  codexPricing?: CodexPricing;
  codexLockdown?: CodexLockdown;
}

export function buildRunConfig(input: BuildRunConfigInput): FrozenConfigCore {
  const harness = input.harness ?? "claude-code";
  return {
    run_type: "mcpatlas_config",
    ratel_version_label: input.ratelVersionLabel,
    ratel_local_version: input.ratelLocalVersion,
    ratel_sdk_version: input.ratelSdkVersion,
    claude_code_version: input.claudeCodeVersion,
    bench_git_sha: input.benchGitSha,
    agent_harness: harness,
    // Conditionally spread, never null-filled: their mere presence on a
    // claude config would change every claude config_hash.
    ...(harness === "codex"
      ? {
          codex_version: input.codexVersion ?? "unknown",
          ...(input.codexPricing ? { codex_pricing: input.codexPricing } : {}),
          ...(input.codexLockdown ? { codex_lockdown: input.codexLockdown } : {}),
        }
      : {}),
    agent_model: input.agentModel,
    backend: null,
    max_turns: input.maxTurns,
    per_cell_timeout_ms: input.perCellTimeoutMs,
    disallowed_tools: [...DISALLOWED_TOOLS],
    permission_mode: input.permissionMode,
    prompt_id: PROMPT_ID,
    prompt_hash: PROMPT_HASH,
    system_prompt_addendum_hash: SYSTEM_PROMPT_ADDENDUM_HASH,
    judge_model: input.judgeModel,
    claim_pass_threshold: DEFAULT_PASS_THRESHOLD,
    claim_partial_threshold: DEFAULT_PARTIAL_THRESHOLD,
    include_tool_evidence: false,
    judge_prompt_sha256: createHash("sha256").update(JUDGE_SYSTEM).digest("hex"),
    retriever_method: input.retrieverMethod,
    top_k_tools: input.topKTools,
    top_k_skills: input.topKSkills,
    arms: input.arms,
    catalogs: [input.manifest],
    eval_ks: input.evalKs,
    catalog_tools: input.catalogTools,
    runs_per_task: input.runsPerTask,
    seed: input.seed,
    concurrency: input.concurrency,
    corpus: {
      name: "mcp-atlas",
      dataset_revision: input.datasetRevision,
      task_list_hash: input.taskListHash,
      task_count: input.taskIds.length,
      task_ids: input.taskIds,
    },
    atlas_sandbox_url: input.sandboxUrl,
    atlas_image_digests: input.atlasImageDigests,
    dollar_cap_global: input.dollarCapGlobal,
    declared_limitations: input.declaredLimitations,
  };
}

export function computeConfigHash(cfg: FrozenConfigCore): string {
  return createHash("sha256").update(stableStringify(cfg)).digest("hex");
}

export function freezeConfig(
  cfg: FrozenConfigCore,
  runId: string,
  generatedAt: string,
): McpAtlasRunConfig {
  return { ...cfg, run_id: runId, generated_at: generatedAt, config_hash: computeConfigHash(cfg) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Queue
// ─────────────────────────────────────────────────────────────────────────────

export interface QueueItem {
  task: McpAtlasTask;
  arm: McpAtlasArm;
  runIndex: number;
}

/** Both arms of a task emitted back-to-back, in a fixed order, before the next
 *  task — so a task's native and ratel cells see the closest possible external
 *  state (live GitHub/Airtable/Mongo data can drift between cells). */
export function buildQueue(
  tasks: readonly McpAtlasTask[],
  arms: readonly McpAtlasArm[],
  runsPerTask: number,
): QueueItem[] {
  const out: QueueItem[] = [];
  for (const task of tasks) {
    for (let runIndex = 0; runIndex < runsPerTask; runIndex++) {
      for (const arm of arms) out.push({ task, arm, runIndex });
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Native-arm cache
// ─────────────────────────────────────────────────────────────────────────────

export interface NativeCacheKeyInput {
  taskId: string;
  model: string;
  scope: McpAtlasScope;
  catalogTools: number;
  runIndex: number;
  agentVersion: string;
  promptHash: string;
  taskListHash: string;
  datasetRevision: string;
  /** Defaulted to claude-code so pre-harness callers and legacy cells key
   *  identically to before. */
  harness?: AgentHarness;
}

/**
 * The native arm never touches ratel-local, so it can be reused across
 * `ratel_local_version` changes — deliberately absent from this key. `scope`
 * IS included: here the catalog size is the tool surface itself, and omitting
 * it would serve a 79-tool cell for a 195-tool cell.
 *
 * `catalogTools` is the same variable expressed continuously, and is included
 * for the same reason. Native's whole context IS the catalog — roughly 11k
 * schema tokens at 40 tools against 24k at 127 — so cells at different sizes
 * are different measurements. Without it a size sweep would serve every native
 * cell from the first size's cache, comparing ratel-at-40 against
 * native-at-127: the gateway would appear to save far more context than it
 * does, with no error raised and a publishable-looking number. (ratel is
 * unaffected — it is never cached.)
 *
 * `harness` is an explicit component for the same reason as `catalogTools`:
 * a codex cell and a claude cell at the same (task, model, scope) are
 * different measurements. Relying on `agentVersion` to differ is not enough —
 * both harnesses report "unknown" under --skip-doctor and would silently
 * cross-serve.
 */
export function nativeCacheKey(input: NativeCacheKeyInput): string {
  return [
    input.taskId,
    "native",
    input.model,
    input.scope,
    input.catalogTools,
    input.runIndex,
    input.agentVersion,
    input.promptHash,
    input.taskListHash,
    input.datasetRevision,
    input.harness ?? "claude-code",
  ].join("::");
}

export interface NativeCacheIndex {
  /** Cells already at the identical key — skip entirely, don't even re-stamp. */
  current: Set<string>;
  /** Cells from any prior write of this key — reusable, re-stamped on write. */
  reuse: Map<string, McpAtlasCell>;
}

/** Scans a prior `agent.jsonl`-shaped file for native-arm cells, keeping the
 *  earliest `generated_at` on a key collision. `current` and `reuse` overlap by
 *  design — a caller who wants literal reuse-without-restamp can check `current`
 *  first. */
/**
 * Indexes prior native cells for reuse, keyed by exactly the same inputs
 * `drainNativeCache`'s caller uses to key the live queue — `promptHash`,
 * `taskListHash`, and `datasetRevision` must be the CURRENT run's values here,
 * not the cell's own recorded ones, or a cache built from a different pinned
 * corpus would collide on task_id/model/scope alone and reuse a stale row.
 */
export function readNativeCacheIndex(
  cells: readonly McpAtlasCell[],
  context: { promptHash: string; taskListHash: string; datasetRevision: string },
): NativeCacheIndex {
  const reuse = new Map<string, McpAtlasCell>();
  const current = new Set<string>();
  for (const c of cells) {
    if (c.arm !== "native") continue;
    const key = nativeCacheKey({
      taskId: c.task_id,
      model: c.model,
      scope: c.catalog_scope,
      // Older rows predate this field; treat as the whole scope.
      catalogTools: c.catalog_tools ?? 0,
      runIndex: c.run_index,
      agentVersion: c.agent_version,
      // Same legacy pattern: rows predating the field are claude-code.
      harness: c.agent_harness ?? "claude-code",
      promptHash: context.promptHash,
      taskListHash: context.taskListHash,
      datasetRevision: context.datasetRevision,
    });
    current.add(key);
    const existing = reuse.get(key);
    if (!existing || c.generated_at < existing.generated_at) reuse.set(key, c);
  }
  return { current, reuse };
}

export interface DrainNativeCacheResult {
  toRun: QueueItem[];
  reusedCells: McpAtlasCell[];
}

/** Removes native-arm queue items that have an exact-key match in the cache
 *  index, re-stamping the reused row onto the current run/config identity.
 *  `--refresh-native` (refreshNative=true) disables reuse entirely. */
export function drainNativeCache(
  queue: readonly QueueItem[],
  keyOf: (item: QueueItem) => string,
  index: ReadonlyMap<string, McpAtlasCell>,
  refreshNative: boolean,
  restamp: (cell: McpAtlasCell, item: QueueItem) => McpAtlasCell,
): DrainNativeCacheResult {
  if (refreshNative) return { toRun: [...queue], reusedCells: [] };
  const toRun: QueueItem[] = [];
  const reusedCells: McpAtlasCell[] = [];
  for (const item of queue) {
    if (item.arm !== "native") {
      toRun.push(item);
      continue;
    }
    const hit = index.get(keyOf(item));
    if (hit) reusedCells.push(restamp(hit, item));
    else toRun.push(item);
  }
  return { toRun, reusedCells };
}

// ─────────────────────────────────────────────────────────────────────────────
// Native-arm latency/token baseline — campaign-wide, per-tool-id
// ─────────────────────────────────────────────────────────────────────────────

/** Median `took_ms` per tool id, over every native tool-call row seen — live or
 *  cache-reused. Feeds `buildLatencyBreakdown`'s `nativeBaselineMs` for ratel
 *  cells; absent entries just mean `gateway_overhead_ms_est` is null for that
 *  tool, which is the correct behavior on a `--arms ratel` only campaign. */
export function collectNativeBaselineMs(rows: readonly McpAtlasToolCallRow[]): Map<string, number> {
  const byTool = new Map<string, number[]>();
  for (const r of rows) {
    if (r.arm !== "native" || r.took_ms == null) continue;
    const list = byTool.get(r.tool_id) ?? [];
    list.push(r.took_ms);
    byTool.set(r.tool_id, list);
  }
  const out = new Map<string, number>();
  for (const [tool, ms] of byTool) {
    const sorted = [...ms].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    out.set(tool, sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch dir per cell
// ─────────────────────────────────────────────────────────────────────────────

export interface CellScratch {
  cellDir: string;
  homeDir: string;
  workspaceDir: string;
  mcpConfigPath: string;
  serveConfigPath: string;
  telemetryPath: string;
  stderrLogPath: string;
}

export function makeScratch(cellKey: string, rootDir: string): CellScratch {
  const cellDir = join(rootDir, cellKey);
  // A prior --keep-artifacts run at the same cell_key leaves its home/session
  // files behind; mkdirSync alone would layer this run's Claude Code session
  // on top of that stale state (multiple .claude/projects/*.jsonl transcripts,
  // stale MCP connection logs) rather than starting clean. Wipe first.
  rmSync(cellDir, { recursive: true, force: true });
  const homeDir = join(cellDir, "home");
  const workspaceDir = join(cellDir, "workspace");
  for (const d of [cellDir, homeDir, workspaceDir]) mkdirSync(d, { recursive: true });
  return {
    cellDir,
    homeDir,
    workspaceDir,
    mcpConfigPath: join(cellDir, "mcp.json"),
    serveConfigPath: join(cellDir, "ratel.json"),
    telemetryPath: join(cellDir, "telemetry.jsonl"),
    stderrLogPath: join(cellDir, "stderr.log"),
  };
}

export function cleanupScratch(scratch: CellScratch, keep: boolean): void {
  if (!keep) rmSync(scratch.cellDir, { recursive: true, force: true });
}

// ─────────────────────────────────────────────────────────────────────────────
// Sandbox lifecycle — plain `docker run`, no compose file exists in this repo
// ─────────────────────────────────────────────────────────────────────────────

export interface SandboxOptions {
  containerName: string;
  image: string;
  port: number;
  envFile: string;
  templateMount?: { hostPath: string; containerPath: string };
}

export interface SandboxHandle {
  containerId: string;
  startedByUs: boolean;
  imageDigest: string;
}

function sh(cmd: string, args: string[]): string {
  return execFileSync(cmd, args, { encoding: "utf8" }).trim();
}

/** Reuses an already-running container with the same name rather than starting
 *  a second one — lets `--sandbox-url` point at a sandbox brought up by hand
 *  (as this session's verification pass did) without mcpatlas-run fighting it
 *  for port 1984. */
export function ensureSandbox(opts: SandboxOptions): SandboxHandle {
  const running = sh("docker", [
    "ps",
    "--filter",
    `name=^${opts.containerName}$`,
    "--format",
    "{{.ID}}",
  ]);
  if (running) {
    const digest = sh("docker", ["image", "inspect", opts.image, "--format", "{{.Id}}"]);
    return { containerId: running, startedByUs: false, imageDigest: digest };
  }
  sh("docker", ["rm", "-f", opts.containerName]).length; // clean any stopped container of the same name
  const args = [
    "run",
    "-d",
    "--name",
    opts.containerName,
    "-p",
    `${opts.port}:1984`,
    "--env-file",
    opts.envFile,
  ];
  if (opts.templateMount) {
    args.push("-v", `${opts.templateMount.hostPath}:${opts.templateMount.containerPath}:ro`);
  }
  args.push(opts.image);
  const containerId = sh("docker", args);
  const digest = sh("docker", ["image", "inspect", opts.image, "--format", "{{.Id}}"]);
  return { containerId, startedByUs: true, imageDigest: digest };
}

export async function waitForHealth(url: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url.replace(/\/+$/, "")}/health`, {
        signal: AbortSignal.timeout(4000),
      });
      if (res.ok) return true;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 5000));
  }
  return false;
}

export function teardownSandbox(handle: SandboxHandle, containerName: string, keep: boolean): void {
  if (keep || !handle.startedByUs) return;
  try {
    sh("docker", ["stop", containerName]);
    sh("docker", ["rm", containerName]);
  } catch {
    // best-effort teardown; a leftover container is a local nuisance, not a
    // correctness problem for the data already written
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-cell execution
// ─────────────────────────────────────────────────────────────────────────────

export interface RunCellDeps {
  runClaude: (o: RunClaudeOpts) => Promise<RunClaudeOutcome>;
  /** The codex twin of runClaude, selected by cfg.agent_harness. REQUIRED for
   *  the same reason fetchSandboxTools is: a unit test must supply a fake and
   *  can never spawn a real process by omission. */
  runCodex: (o: RunCodexOpts) => Promise<RunClaudeOutcome>;
  judgeClaims: typeof judgeClaims;
  /** Live catalog probe for the per-cell integrity check. REQUIRED (not
   *  defaulted per-field) so a unit test must supply a fake and can never make
   *  a real network call by omission. */
  fetchSandboxTools: (url: string) => Promise<AtlasTool[] | null>;
}

export const REAL_RUN_CELL_DEPS: RunCellDeps = {
  runClaude: realRunClaude,
  runCodex: realRunCodex,
  judgeClaims,
  fetchSandboxTools,
};

export interface RunCellOptions {
  item: QueueItem;
  cfg: McpAtlasRunConfig;
  manifest: McpAtlasCatalogManifest;
  shim: ShimSpec;
  scratchRoot: string;
  keepArtifacts: boolean;
  judgeModel?: LanguageModel;
  nativeBaselineMs: Map<string, number>;
  nativeCatalogTokens: number;
  gatewaySchemaTokens: number;
  perToolTokens?: Map<string, number>;
  cacheSource: "live" | "reused";
  deps?: RunCellDeps;
}

export interface RunCellResult {
  cell: McpAtlasCell;
  toolCallRows: McpAtlasToolCallRow[];
  searchEventRows: McpAtlasSearchEventRow[];
  retrievalRows: McpAtlasRetrievalRow[];
  dollarCost: number;
}

/** Scratch-dir and row identity for one cell. Harness enters only as a
 *  codex-only suffix: claude cell keys are persisted run output and must stay
 *  byte-identical, while two harnesses at the same (task, arm, scope, run)
 *  must not share a scratch dir (`makeScratch` rmSyncs it). */
export function cellKeyFor(
  item: QueueItem,
  scope: McpAtlasScope,
  harness: AgentHarness = "claude-code",
): string {
  const base = `${item.task.task_id}__${item.arm}__s${scope}__r${item.runIndex}`;
  return harness === "codex" ? `${base}__hcodex` : base;
}

/** `process.env`, stripped of undefined entries. Passed to the spawned `claude`
 *  process so it inherits PATH (to find the `claude`/`npx`/`node` binaries) and
 *  whatever auth it needs — an empty env makes `spawn` fail with ENOENT before
 *  any API call happens, which looks like a $0 cell error with no useful
 *  signal about what actually broke. */
function inheritedEnv(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) if (v !== undefined) out[k] = v;
  return out;
}

/** Shared, run-independent cache for ratel-local's embedding model.
 *
 *  Semantic/hybrid retrieval loads BAAI/bge-small-en-v1.5 (128 MB) into
 *  ratel-local. The HuggingFace cache lives under `$HOME/.cache/huggingface`,
 *  and every cell gets a throwaway HOME for session isolation — so without
 *  this, each of the 30 cells in a k=3 run re-downloads 128 MB and pays a
 *  ~29s model load, against Claude Code's 30s MCP connection timeout. The
 *  gateway never finishes starting, serves zero tools, and the agent falls
 *  back to disallowed built-ins. Measured directly: cold start times out,
 *  warm start reaches `[ratel] ready` with all four tools and no delay.
 *
 *  Pointing HF_HOME at a stable path keeps the isolation that matters (Claude
 *  Code's own session state stays per-cell) while sharing an immutable model
 *  cache, which is not session state. Harmless under bm25, which never loads
 *  an embedding model.
 *
 *  Note this deliberately does NOT inherit the developer's real
 *  ~/.cache/huggingface: a stale token there is sent on download and rejected
 *  with a 401, where anonymous access succeeds. */
export const SHARED_HF_HOME = "results/raw/mcpatlas/.hf-cache";

function embeddingCacheEnv(): Record<string, string> {
  const dir = resolveRepoPath(SHARED_HF_HOME);
  mkdirSync(dir, { recursive: true });
  return { HF_HOME: dir };
}

/** Claude Code's MCP server startup timeout, in ms. Its default is 30s, and
 *  ratel-local under semantic retrieval needs ~44s to reach `[ratel] ready`
 *  even with a warm model cache — measured directly. Under the default the
 *  gateway is killed mid-startup with CONNECT_TIMEOUT, serves zero tools, and
 *  the agent falls back to disallowed built-ins, which is what made every
 *  semantic run void.
 *
 *  Set on the PARENT claude process (a Claude Code setting), unlike HF_HOME
 *  which must go in mcp.json on the server entry itself.
 *
 *  120s is ~2.7x the measured startup with headroom, while staying well inside
 *  the 300s per-cell budget so a genuinely hung server still fails the cell
 *  rather than consuming it. Harmless under bm25, which starts in seconds. */
export const MCP_STARTUP_TIMEOUT_MS = 120_000;

/** The catalog's tool definitions, straight from the sandbox — the manifest
 *  carries only tool ids, and pricing occupancy needs the schemas themselves.
 *  Same endpoint and shape handling as the doctor's `sandboxTools` probe. */
async function fetchSandboxTools(url: string): Promise<AtlasTool[] | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 20_000);
  try {
    const res = await fetch(`${url.replace(/\/+$/, "")}/list-tools`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      signal: ctrl.signal,
    });
    if (!res.ok) return null;
    const body = await res.json();
    if (Array.isArray(body)) return body as AtlasTool[];
    const tools = (body as { tools?: unknown }).tools;
    return Array.isArray(tools) ? (tools as AtlasTool[]) : null;
  } catch {
    return null;
  } finally {
    clearTimeout(t);
  }
}

/**
 * Which of `toolIds` the live sandbox is NOT currently serving.
 *
 * The doctor asserts the catalog once at campaign start and nothing ever looks
 * again — but upstream servers inside the sandbox die independently of the
 * container (measured: desktop-commander flapping since 08/24 took 21 tools and
 * left 10 of 55 tasks silently unsolvable, scoring as retrieval failures).
 * `gold_coverage` cannot catch it: it is computed from the manifest, not from
 * what the sandbox actually serves, so it reads 1.0 either way.
 *
 * Sandbox names are `<server>_<bare>`; canonicalise before comparing.
 */
export function missingFromSandbox(
  sandboxTools: readonly AtlasTool[],
  toolIds: readonly string[],
  knownServers: readonly string[],
): string[] {
  const served = new Set(sandboxTools.map((t) => normalizeToolId(t.name, knownServers) ?? t.name));
  return toolIds.filter((id) => !served.has(id));
}

/**
 * Servers whose ratel-local registration disagrees with the manifest — either
 * absent from telemetry entirely (the server never came up behind the gateway)
 * or registered with a different tool count (partial drift). The complement of
 * `missingFromSandbox` for the ratel arm: that check proves the SANDBOX was
 * healthy before the cell; this one proves the GATEWAY actually registered
 * what the manifest says, from the registration telemetry the cell itself
 * produced.
 */
export function registrationMismatches(
  telemetryText: string,
  manifest: McpAtlasCatalogManifest,
): string[] {
  const counts = new Map<string, number>();
  for (const e of parseTelemetry(telemetryText)) {
    if (e.type === "ratel_tool_payload") {
      const p = e as { server?: string; tool_count?: number };
      if (typeof p.server === "string") counts.set(p.server, Number(p.tool_count) || 0);
    }
  }
  // No payload events at all: an old ratel-local that predates the event, or a
  // telemetry format change. Refusing here would fail every cell of such a
  // version, so leave that failure mode to the empty-telemetry check.
  if (counts.size === 0) return [];
  const out: string[] = [];
  for (const srv of manifest.servers) {
    const got = counts.get(srv.server);
    if (got === undefined) out.push(`${srv.server}: not registered`);
    else if (got !== srv.tool_count) {
      out.push(`${srv.server}: registered ${got} tools, manifest has ${srv.tool_count}`);
    }
  }
  return out;
}

/**
 * Ground truth for both arms' `tool_schema_tokens`, measured ONCE per campaign
 * before any cell runs.
 *
 * WHY THIS EXISTS AT ALL. These two numbers used to be threaded in as literal
 * `0` and then read back off the cells they had been written to — a loop that
 * could never resolve, so `tool_schema_tokens` and `schema_share_of_prefix`
 * were 0 on every cell of both arms in every run to date. That is the headline
 * occupancy metric: the whole "127 schemas become 4" claim is this field.
 *
 * WHY IT CANNOT COME FROM TELEMETRY. ratel-local emits `ratel_tool_payload`
 * registration events carrying its own `estimated_tokens`, and those are
 * genuinely good — 25,759 for the coding catalog against an observed 24,343
 * real-token first-turn difference. But it emits nothing for its OWN four
 * gateway tools, so taking the native side from telemetry and the gateway side
 * from `schemaTokenEstimate` would put the two arms on different rulers and
 * silently inflate the savings by the ~7% they disagree on. Both sides are
 * measured here with one function instead.
 *
 * The gateway side is asked of the gateway directly rather than derived from
 * `GATEWAY_TOOLS`, because that constant lists what the agent is ALLOWED to
 * call (2 names) while context is occupied by everything `tools/list` returns
 * (4, including `search_capabilities` and `auth`). Occupancy must be measured
 * from what is actually in the prompt.
 *
 * Returns null if the gateway cannot be reached; the caller decides whether
 * that is fatal, and records zero rather than a guess.
 */
export async function measureGatewaySchemaTokens(o: {
  manifest: McpAtlasCatalogManifest;
  shim: ShimSpec;
  retrieverMethod: "bm25" | "semantic" | "hybrid";
  ratelLocalPin: string;
  scratchRoot: string;
}): Promise<number | null> {
  const { spawn } = await import("node:child_process");
  const dir = join(o.scratchRoot, ".schema-probe");
  mkdirSync(dir, { recursive: true });
  const cfgPath = join(dir, "ratel.json");
  writeFileSync(
    cfgPath,
    JSON.stringify(buildRatelServeConfig(o.manifest, o.shim, o.retrieverMethod)),
  );

  return await new Promise<number | null>((resolveP) => {
    const child = spawn(
      "npx",
      ["-y", `@ratel-ai/ratel-local@${o.ratelLocalPin}`, "serve", cfgPath],
      {
        stdio: ["pipe", "pipe", "ignore"],
        env: { ...inheritedEnv(), ...embeddingCacheEnv() },
        // Own process group, for the same reason runClaude uses one: `npx` is a
        // launcher, and killing it leaves ratel-local AND its 11 upstream stdio
        // shims running. They hold their pipes open, which keeps this process's
        // event loop alive — the campaign finishes, prints its `done:` line, and
        // then hangs forever instead of exiting.
        detached: true,
      },
    );
    let buf = "";
    let done = false;
    let instructionsTokens = 0;
    const finish = (v: number | null) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      // Signal the group, not the launcher. Fall back to the direct kill if the
      // group is already gone.
      try {
        if (child.pid) process.kill(-child.pid, "SIGKILL");
        else child.kill("SIGKILL");
      } catch {
        try {
          child.kill("SIGKILL");
        } catch {
          /* already gone */
        }
      }
      // Belt and braces: even after the group dies, an inherited pipe can hold
      // the loop open. Nothing reads these again once we have the answer.
      child.stdout?.destroy();
      child.stdin?.destroy();
      child.unref();
      resolveP(v);
    };
    // Semantic startup downloads/loads an embedding model; MCP_STARTUP_TIMEOUT_MS
    // is the same budget a real cell gives it.
    const timer = setTimeout(() => finish(null), MCP_STARTUP_TIMEOUT_MS);

    child.on("error", () => finish(null));
    child.on("close", () => finish(null));
    child.stdout.on("data", (d) => {
      buf += d;
      let nl = buf.indexOf("\n");
      while (nl >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        nl = buf.indexOf("\n");
        if (!line) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.id === 1) {
            // The gateway ships a server `instructions` block ("call
            // search_capabilities first...", plus a per-server tool census).
            // It lands in the ratel arm's context and the native arm never
            // pays it, so leaving it out would overstate the gateway's
            // savings. It is occupancy, so it counts.
            const instr = msg.result?.instructions;
            instructionsTokens = typeof instr === "string" ? Math.ceil(instr.length / 4) : 0;
            child.stdin.write(
              `${JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })}\n`,
            );
            child.stdin.write(
              `${JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list", params: {} })}\n`,
            );
          } else if (msg.id === 2) {
            const tools = msg.result?.tools;
            finish(Array.isArray(tools) ? schemaTokenEstimate(tools) + instructionsTokens : null);
          }
        } catch {
          // ratel-local writes only JSON-RPC to stdout, but be tolerant.
        }
      }
    });

    child.stdin.write(
      `${JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2025-06-18",
          capabilities: {},
          clientInfo: { name: "mcpatlas-schema-probe", version: "1" },
        },
      })}\n`,
    );
  });
}

/** Runs one (task, arm) cell end to end: writes the cell's mcp.json/ratel.json,
 *  invokes Claude Code, reads the transcript and telemetry, judges the claims,
 *  and assembles all four row types. On any failure — including the case that
 *  matters most, an empty telemetry file on a `ratel` cell — returns a
 *  complete, schema-valid result with `cell.error` set, never a missing row. */
export async function runCell(o: RunCellOptions): Promise<RunCellResult> {
  const deps = o.deps ?? REAL_RUN_CELL_DEPS;
  const { item, cfg } = o;
  // Per-task catalog when a size sweep is active: this task's gold plus seeded
  // fillers from the scope. Seeded on (task, size, seed) so a rerun reproduces
  // it and BOTH ARMS get the identical set — catalog_sha256 is asserted equal
  // across arms and that assertion is what makes the comparison valid.
  const subsetting = cfg.catalog_tools > 0;
  const manifest = subsetting
    ? subsetManifest(
        o.manifest,
        selectCatalogTools(
          o.manifest,
          item.task.gold_tool_ids,
          cfg.catalog_tools,
          `${item.task.task_id}|${cfg.catalog_tools}|${cfg.seed}`,
        ),
      )
    : o.manifest;
  // Occupancy must follow the catalog the agent actually saw. The campaign-wide
  // nativeCatalogTokens prices all 127 tools; a subset costs less, and reporting
  // the full figure would overstate native's context and inflate the savings.
  const nativeCatalogTokens =
    subsetting && o.perToolTokens
      ? manifest.servers
          .flatMap((s) => s.tool_ids)
          .reduce((n, id) => n + (o.perToolTokens?.get(id) ?? 0), 0)
      : o.nativeCatalogTokens;
  const cellKey = cellKeyFor(item, cfg.catalogs[0].scope, cfg.agent_harness);
  const scratch = makeScratch(cellKey, o.scratchRoot);
  const knownServers = manifest.servers.map((s) => s.server);
  const isCodex = cfg.agent_harness === "codex";
  const codexHomeDir = join(scratch.homeDir, ".codex");

  try {
    // Catalog integrity, per cell, BEFORE any spend. The doctor asserts this
    // once at campaign start; upstream servers inside the sandbox die
    // independently of the container (desktop-commander did, mid-campaign,
    // taking 21 tools and leaving 10/55 tasks silently unsolvable). A cell run
    // against an incomplete catalog scores as a retrieval failure — the
    // pool-contamination mode — so refuse the cell instead: the thrown error
    // becomes a schema-valid row with `error` set, and no tokens are bought.
    const sandboxTools = await deps.fetchSandboxTools(o.shim.sandboxUrl);
    if (!sandboxTools) {
      throw new Error(`catalog integrity: sandbox unreachable at ${o.shim.sandboxUrl}`);
    }
    const missing = missingFromSandbox(
      sandboxTools,
      manifest.servers.flatMap((s) => s.tool_ids),
      // Normalise against the FULL scope's server names — under subsetting the
      // per-task manifest may have dropped a server whose prefix still appears
      // in sandbox tool names.
      o.manifest.servers.map((s) => s.server),
    );
    if (missing.length > 0) {
      throw new Error(
        `catalog integrity: sandbox no longer serves ${missing.length} tool(s) in this cell's ` +
          `catalog: ${missing.slice(0, 8).join(", ")}${missing.length > 8 ? ", ..." : ""}`,
      );
    }

    if (!isCodex) {
      if (item.arm === "native") {
        writeFileSync(
          scratch.mcpConfigPath,
          JSON.stringify(buildNativeMcpConfig(manifest, o.shim, subsetting)),
        );
      } else {
        writeFileSync(
          scratch.serveConfigPath,
          JSON.stringify(buildRatelServeConfig(manifest, o.shim, cfg.retriever_method, subsetting)),
        );
        writeFileSync(
          scratch.mcpConfigPath,
          JSON.stringify(
            buildRatelMcpConfig({
              ratelLocalPin: cfg.ratel_local_version,
              serveConfigPath: scratch.serveConfigPath,
              telemetryPath: scratch.telemetryPath,
              // Must go in mcp.json, not just the parent process env: Claude Code
              // does not propagate its own environment to spawned stdio MCP
              // servers, so HF_HOME set on `claude` never reached ratel-local and
              // the embedding cache stayed cold (verified: a full semantic k=3 run
              // still produced zero searches with HF_HOME set only on the parent).
              // PATH is included because a server-level `env` may replace rather
              // than merge, and npx needs it.
              env: { ...embeddingCacheEnv(), PATH: process.env.PATH ?? "" },
            }),
          ),
        );
      }
    } else {
      // Codex takes its MCP config from $CODEX_HOME/config.toml rather than a
      // per-invocation flag. The entries are rendered from the SAME builders
      // as the claude mcp.json — including ratel.json, which is ratel-local's
      // own config and must be byte-identical across harnesses — so the arms
      // differ across harnesses only in host-config serialization.
      let mcpServers: ReturnType<typeof buildNativeMcpConfig>["mcpServers"];
      if (item.arm === "native") {
        mcpServers = buildNativeMcpConfig(manifest, o.shim, subsetting).mcpServers;
      } else {
        writeFileSync(
          scratch.serveConfigPath,
          JSON.stringify(buildRatelServeConfig(manifest, o.shim, cfg.retriever_method, subsetting)),
        );
        mcpServers = buildRatelMcpConfig({
          ratelLocalPin: cfg.ratel_local_version,
          serveConfigPath: scratch.serveConfigPath,
          telemetryPath: scratch.telemetryPath,
          // Same env lesson as the claude mcp.json above: HF_HOME must ride on
          // the server entry itself, and PATH because a server-level env may
          // replace rather than merge.
          env: { ...embeddingCacheEnv(), PATH: process.env.PATH ?? "" },
        }).mcpServers;
      }
      mkdirSync(codexHomeDir, { recursive: true });
      writeFileSync(
        join(codexHomeDir, "config.toml"),
        buildCodexConfigToml({
          model: cfg.agent_model,
          mcpServers,
          // The claude driver passes the same budget via the MCP_TIMEOUT env
          // var; Codex takes it per-server as startup_timeout_sec.
          startupTimeoutSec: MCP_STARTUP_TIMEOUT_MS / 1000,
          toolTimeoutSec: cfg.codex_lockdown?.tool_timeout_sec ?? CODEX_TOOL_TIMEOUT_SEC,
          // Only the ratel gateway is required — it IS the tool surface and is
          // npx-slow to start. Native shims stay optional so one flaky shim
          // does not abort the session. See buildCodexConfigToml.
          requiredServers: item.arm === "ratel" ? ["ratel-local"] : [],
        }),
      );
      // Codex has no --append-system-prompt. The addendum arrives as the
      // workspace's AGENTS.md project doc instead — same frozen text, hashed
      // into system_prompt_addendum_hash; the delivery difference is recorded
      // in codex_lockdown.addendum_delivery.
      writeFileSync(join(scratch.workspaceDir, "AGENTS.md"), SYSTEM_PROMPT_ADDENDUM);
    }

    const allowedTools =
      item.arm === "native"
        ? manifest.servers.flatMap((s) =>
            s.tool_ids.map((id) => `mcp__${id.split("/")[0]}__${id.split("/").slice(1).join("/")}`),
          )
        : GATEWAY_TOOLS.map((t) => `mcp__ratel-local__${t}`);

    const outcome = isCodex
      ? await deps.runCodex({
          args: buildCodexArgs({
            prompt: buildPrompt(item.task.prompt),
            cwd: scratch.workspaceDir,
          }),
          cwd: scratch.workspaceDir,
          codexHome: codexHomeDir,
          env: { ...inheritedEnv(), ...embeddingCacheEnv() },
          timeoutMs: cfg.per_cell_timeout_ms,
        })
      : await deps.runClaude({
          prompt: buildPrompt(item.task.prompt),
          mcpConfigPath: scratch.mcpConfigPath,
          allowedTools,
          model: cfg.agent_model,
          maxTurns: cfg.max_turns,
          permissionMode: cfg.permission_mode,
          cwd: scratch.workspaceDir,
          homeDir: scratch.homeDir,
          env: {
            ...inheritedEnv(),
            ...embeddingCacheEnv(),
            MCP_TIMEOUT: String(MCP_STARTUP_TIMEOUT_MS),
          },
          timeoutMs: cfg.per_cell_timeout_ms,
          appendSystemPrompt: SYSTEM_PROMPT_ADDENDUM,
        });

    let result: ClaudeResult;
    let tPath: string | null;
    let parsed: ParsedTranscript | undefined;
    let codexEvents: ReturnType<typeof parseCodexEvents> = null;
    if (isCodex) {
      const events = parseCodexEvents(outcome.stdout);
      if (!events) {
        throw new Error(
          `codex produced no parseable events (timedOut=${outcome.timedOut}, exitCode=${outcome.exitCode}, signal=${outcome.signal}): ${outcome.stderr.slice(0, 500)}`,
        );
      }
      codexEvents = events;
      const pricing = cfg.codex_pricing;
      if (!pricing) {
        throw new Error("codex run config is missing codex_pricing — cannot compute cell cost");
      }
      result = codexResultFromEvents(events, outcome.wallMs, outcome.timedOut, pricing);
      // The rollout file is the codex "transcript": provenance, and the only
      // place compaction is visible (the --json stream carries no signal).
      tPath = codexRolloutPath(codexHomeDir, events.threadId);
      parsed = {
        uses: events.uses,
        turnUsages: events.turnUsages,
        compactionEvents: countCodexCompactions(readTranscript(tPath)),
        reasoningOutputTokens: events.reasoningOutputTokens,
      };
    } else {
      const claudeResult = parseClaudeResult(outcome.stdout);
      if (!claudeResult) {
        throw new Error(
          `claude produced no parseable result envelope (timedOut=${outcome.timedOut}, exitCode=${outcome.exitCode}, signal=${outcome.signal}): ${outcome.stderr.slice(0, 500)}`,
        );
      }
      result = claudeResult;
      tPath = transcriptPath(scratch.homeDir, scratch.workspaceDir, result.session_id);
    }
    const transcriptText = readTranscript(tPath);
    const telemetryText =
      item.arm === "ratel" && existsSync(scratch.telemetryPath)
        ? readFileSync(scratch.telemetryPath, "utf8")
        : "";

    if (item.arm === "ratel" && telemetryText.trim().length === 0) {
      // Hard row-level error, not a warning — an empty telemetry file on a
      // ratel cell means the retrieval measurement for this cell is dead.
      throw new Error("ratel cell produced empty telemetry — retrieval data for this cell is lost");
    }

    if (item.arm === "ratel") {
      // The sandbox probe above proved the SANDBOX was healthy before the
      // cell; this proves the GATEWAY registered what the manifest says, from
      // the cell's own registration telemetry. A server that failed behind
      // ratel-local mid-cell shows up here and nowhere else.
      const regProblems = registrationMismatches(telemetryText, manifest);
      if (regProblems.length > 0) {
        throw new Error(
          `catalog integrity: ratel-local registration disagrees with the manifest — ` +
            regProblems.join("; "),
        );
      }
    }

    const ctx: CellContext = {
      run_id: cfg.run_id,
      config_hash: cfg.config_hash,
      generated_at: cfg.generated_at,
      cell_key: cellKey,
      task: item.task,
      arm: item.arm,
      catalog_scope: cfg.catalogs[0].scope,
      catalog_tools: cfg.catalog_tools,
      catalog_tool_ids: manifest.servers.flatMap((s) => s.tool_ids),
      eval_ks: cfg.eval_ks,
      per_call_timeout_ms: cfg.per_cell_timeout_ms,
      model: cfg.agent_model,
      ratel_version_label: cfg.ratel_version_label,
      ratel_local_version: cfg.ratel_local_version,
      ratel_sdk_version: cfg.ratel_sdk_version,
      agent_harness: cfg.agent_harness,
    };

    const claimRubric = await deps.judgeClaims({
      taskId: item.task.task_id,
      prompt: item.task.prompt,
      claims: item.task.claims,
      finalText: result.result,
      model: o.judgeModel,
    });

    // The native codex arm has no ratel telemetry, so its tool-call
    // success/failure comes from codex's own event stream instead — otherwise
    // every call would default to "ok" and mask denials/upstream errors.
    const codexNativeSpans =
      codexEvents && item.arm === "native"
        ? codexInvokeSpans(codexEvents, knownServers)
        : undefined;

    const cell = assembleCell({
      ctx,
      result,
      transcriptText,
      transcriptPath: tPath ?? "",
      telemetryText,
      telemetryPath: item.arm === "ratel" ? scratch.telemetryPath : null,
      claimRubric,
      nativeCatalogTokens,
      gatewaySchemaTokens: o.gatewaySchemaTokens,
      perToolTokens: o.perToolTokens,
      nativeBaselineMs: o.nativeBaselineMs,
      agentVersion: isCodex ? (cfg.codex_version ?? "unknown") : cfg.claude_code_version,
      runIndex: item.runIndex,
      cacheSource: o.cacheSource,
      ...(parsed ? { parsed, costSource: "computed" as const } : {}),
      ...(codexNativeSpans ? { invokeSpansOverride: codexNativeSpans } : {}),
      ...(codexEvents ? { shellCommandExecutions: codexEvents.commandExecutions } : {}),
    });

    const uses = parsed?.uses ?? parseUses(transcriptText);
    const { calls, offCatalog } = effectiveCalls(uses, knownServers);
    const spans = codexNativeSpans ?? invokeSpans(parseTelemetry(telemetryText), knownServers);
    const toolCallRows = buildToolCallRows(ctx, calls, offCatalog, spans);

    let searchEventRows: McpAtlasSearchEventRow[] = [];
    let retrievalRows: McpAtlasRetrievalRow[] = [];
    if (item.arm === "ratel") {
      const invokedIds = calls.map((c) => c.tool_id);
      const built = buildSearchEventRows(ctx, telemetryText, knownServers, invokedIds);
      searchEventRows = built.rows;
      const queries = built.rows.map((r) => r.query);
      retrievalRows = buildRetrievalRows(
        ctx,
        built.results,
        queries,
        cfg.ratel_local_version,
        cfg.ratel_version_label,
        cfg.retriever_method,
      ).map((r) => ({ ...r, model: cfg.agent_model }));
    }

    cleanupScratch(scratch, o.keepArtifacts);
    return {
      cell,
      toolCallRows,
      searchEventRows,
      retrievalRows,
      dollarCost: result.total_cost_usd ?? 0,
    };
  } catch (err) {
    cleanupScratch(scratch, o.keepArtifacts);
    const message = err instanceof Error ? err.message : String(err);
    const catalogSet = new Set(manifest.servers.flatMap((s) => s.tool_ids));
    const retrievable = item.task.gold_tool_ids.filter((g) => catalogSet.has(g));
    const errorCell: McpAtlasCell = {
      run_type: "mcpatlas_task",
      run_id: cfg.run_id,
      config_hash: cfg.config_hash,
      generated_at: cfg.generated_at,
      cell_key: cellKey,
      task_id: item.task.task_id,
      scenario_id: item.task.id,
      category: "mcpatlas-coding",
      arm: item.arm,
      catalog_scope: cfg.catalogs[0].scope,
      catalog_tool_count: manifest.tool_count,
      catalog_tools: cfg.catalog_tools,
      catalog_size: item.arm === "native" ? manifest.tool_count : GATEWAY_TOOLS.length,
      run_index: item.runIndex,
      ratel_version_label: cfg.ratel_version_label,
      ratel_local_version: cfg.ratel_local_version,
      ratel_sdk_version: cfg.ratel_sdk_version,
      agent_version:
        cfg.agent_harness === "codex" ? (cfg.codex_version ?? "unknown") : cfg.claude_code_version,
      agent_harness: cfg.agent_harness,
      model: cfg.agent_model,
      enabled_tool_ids: item.task.enabled_tool_ids,
      gold_tool_ids: item.task.gold_tool_ids,
      retrievable_gold_ids: retrievable,
      gold_coverage: item.task.gold_tool_ids.length
        ? retrievable.length / item.task.gold_tool_ids.length
        : 1,
      observed_tool_ids: [],
      tool_calls: [],
      claim_rubric: {
        claims: [],
        coverage: null,
        verdict: "n/a",
        judge_model: cfg.judge_model,
        judge_error: message,
        judge_wall_ms: 0,
        judge_input_tokens: 0,
        judge_output_tokens: 0,
      },
      task_pass: false,
      programmatic_verdict: "n/a",
      judge_verdict: "n/a",
      tool_selection_recall: 0,
      tool_selection_precision: 0,
      tool_selection_f1: 0,
      tool_selection_pass: false,
      tool_selection_hit: false,
      trajectory_order_similarity: 0,
      missing_gold: item.task.gold_tool_ids,
      extra_calls: [],
      off_catalog_calls: [],
      tokens: {
        tool_schema_tokens: 0,
        system_prompt_tokens: 0,
        first_turn_context_tokens: 0,
        peak_context_tokens: 0,
        compaction_events: 0,
        retrieval_overhead_tokens: 0,
        tool_result_tokens: 0,
        schema_share_of_prefix: 0,
        billed_input_tokens: 0,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        dollar_cost_total: 0,
        cache_hit_ratio: 0,
      },
      latency: {
        total_ms: 0,
        search_ms_total: 0,
        search_ms_p50: null,
        search_ms_p90: null,
        search_stage_ms: {},
        invoke_ms_total: 0,
        invoke_ms_p50: null,
        gateway_overhead_ms_est: null,
        model_ms_est: 0,
        turns: 0,
      },
      tool_failures: {
        ok: 0,
        upstream_error: 0,
        tool_not_found: 0,
        schema_validation_error: 0,
        auth_error: 0,
        rate_limited: 0,
        timeout: 0,
        transport_error: 0,
        gateway_error: 0,
        oversized_result: 0,
        unknown_error: 0,
        off_catalog_call: 0,
      },
      tool_calls_total: 0,
      tool_calls_unique: 0,
      gateway_calls: 0,
      non_gateway_calls: 0,
      search_count: 0,
      final_text: "",
      finish_reason: "error",
      error: message,
      transcript_path: "",
      telemetry_path: null,
      telemetry_binding: "none",
      cache_source: o.cacheSource,
    };
    return {
      cell: errorCell,
      toolCallRows: [],
      searchEventRows: [],
      retrievalRows: [],
      dollarCost: 0,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker pool
// ─────────────────────────────────────────────────────────────────────────────

export interface CampaignOptions<T> {
  concurrency: number;
  dollarCap: number | null;
  onCell: (r: RunCellResult) => void;
  runOne: (item: T) => Promise<RunCellResult>;
  quiet?: boolean;
}

export interface CampaignSummary {
  cells_run: number;
  cells_skipped: number;
  total_dollars: number;
  stopped_reason: "completed" | "global_cap";
}

/** Hand-rolled worker pool mirroring agent/src/sragents-select.ts's `worker()`
 *  loop. Best-effort dollar cap: overshoot is bounded by ~concurrency in-flight
 *  cells, since the check only runs at the top of each worker's next iteration
 *  — an accepted tradeoff, not a bug. Generic over the queue item type: the
 *  pool never inspects an item itself, only dispatches it to `runOne`. */
export async function runCampaign<T>(
  queue: readonly T[],
  opts: CampaignOptions<T>,
): Promise<CampaignSummary> {
  let i = 0;
  let dollars = 0;
  let stopped = false;
  let run = 0;

  async function worker(): Promise<void> {
    while (true) {
      if (opts.dollarCap != null && dollars >= opts.dollarCap) {
        stopped = true;
        return;
      }
      const idx = i++;
      if (idx >= queue.length) return;
      const result = await opts.runOne(queue[idx]);
      dollars += result.dollarCost;
      run++;
      opts.onCell(result);
      if (!opts.quiet) {
        console.log(
          `[${run}/${queue.length}] ${result.cell.cell_key} ${result.cell.error ? "ERROR" : result.cell.task_pass ? "pass" : "fail"} $${dollars.toFixed(4)}`,
        );
      }
    }
  }

  await Promise.all(Array.from({ length: Math.max(1, opts.concurrency) }, () => worker()));

  return {
    cells_run: run,
    cells_skipped: queue.length - run,
    total_dollars: dollars,
    stopped_reason: stopped ? "global_cap" : "completed",
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

export function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

export function formatDoneLine(summary: CampaignSummary, cellsCached: number): string {
  return (
    `done: ${summary.cells_run} cells run, ${cellsCached} cached, ` +
    `${summary.cells_skipped} skipped, $${summary.total_dollars.toFixed(4)} spent, ` +
    `stopped=${summary.stopped_reason}`
  );
}

async function loadLocalEnv(): Promise<void> {
  const { config } = await import("dotenv");
  const here = new URL("..", import.meta.url).pathname;
  config({ path: `${here}.env`, quiet: true });
}

export async function main(): Promise<void> {
  await loadLocalEnv();

  const { runChecks, defaultProbes, doctorExitCode, formatResults } = await import(
    "./mcpatlas-doctor.js"
  );
  type ResolvedFactsShape = Awaited<ReturnType<typeof runChecks>>["facts"];

  const scope = arg("--scope", "coding") as McpAtlasScope;
  const armsFlag = arg("--arms", "native,ratel");
  const arms = armsFlag.split(",").map((a) => a.trim()) as McpAtlasArm[];
  const tasksLimit = Number(arg("--tasks", "0"));
  const tasksOffset = Number(arg("--task-offset", "0"));
  const concurrency = Number(arg("--concurrency", "1"));
  const dollarCap = Number(arg("--dollar-global", "50"));
  const refreshNative = process.argv.includes("--refresh-native");
  const keepStack = process.argv.includes("--keep-stack");
  const keepArtifacts = process.argv.includes("--keep-artifacts");
  const skipDoctor = process.argv.includes("--skip-doctor");
  const force = process.argv.includes("--force");
  const cacheSourcePath = arg("--cache-source", "");
  const outputPath = resolveRepoPath(arg("--output", "results/raw/mcpatlas/agent.jsonl"));
  const sandboxUrl = arg("--sandbox-url", process.env.MCP_SANDBOX_URL ?? "http://localhost:1984");
  const ratelLocalPin = arg("--ratel-local", process.env.RATEL_LOCAL_VERSION ?? "0.8.1");
  const model = arg("--model", process.env.RATEL_BENCH_MODEL ?? "claude-haiku-4-5");
  const judgeModelId = arg("--judge-model", "");
  // k=1 by default, and this is a measured decision rather than a cost-saving
  // default — do not raise it for the headline run without re-deriving.
  //
  // k=1 makes every per-task result a single sample, so run-to-run swings are
  // indistinguishable from real effects: the `k1_no_variance` limitation. A
  // 5-task x 2-arm x 3-run smoke measured how large that actually is — 2 of 5
  // tasks flipped on each arm (native P.P / .P., ratel P.P / P..), and had we
  // run once the ratel headline would have read 40%, 0% or 20% depending on
  // which single run we drew.
  //
  // That looks like an argument for k=3 and is not, because it conflates two
  // noise sources. Only one of them scales with k:
  //
  //   Var(mean) = (1/n) [ Var(p_i)  +  E[p_i(1-p_i)] / k ]
  //                       ^^^^^^^^     ^^^^^^^^^^^^^^^^^
  //                       which tasks  run-to-run flips
  //                       you chose    (the only k term)
  //
  // At n=5 one flip moves the headline 20pp, which is why the smoke swung so
  // hard. At the real n=55 one flip moves it 1.8pp, and task sampling
  // dominates — a floor k cannot lower. Substituting the observed flip rate
  // gives roughly +/-12pp per arm at k=1 against +/-10pp at k=3: triple the
  // cells and triple the spend to buy about 2pp, on a delta that is nowhere
  // near significant at either k.
  //
  // So the honest way to spend the k budget is not a third run of all 55
  // tasks but `--variance-subset`: a handful of tasks at higher k, enough to
  // report a measured `success_flip_rate` instead of the bare
  // `k1_no_variance` caveat. Those rows carry run_index > 0 and are excluded
  // from headline aggregation, so they never over-weight the subset.
  //
  // Cells stay distinct because run_index is part of cell_key and of the
  // native cache key, so raising this for a targeted question is always safe.
  // Catalog-size sweep. 0 (default) = whole scope, the historical behaviour.
  // Any positive value gives each task a catalog of its gold tools plus seeded
  // fillers from the scope, so gold coverage stays complete at every size and
  // the gold fraction stays roughly constant across tasks.
  const catalogTools = Math.max(0, Number(arg("--catalog-tools", "0")));
  const runsPerTask = Math.max(1, Number(arg("--runs", "1")));
  // Upstream MCP-Atlas defaults (services/agent-harness: DEFAULT_MAX_TURNS=256,
  // README: --timeout 1800s). Ours were hardcoded at 20 turns / 300s — the 20
  // was an arbitrary value, and at k=3 it terminated 5 of 15 ratel cells while
  // never once binding on native, biasing the arm comparison against the arm
  // whose search->invoke indirection legitimately costs more turns.
  //
  // Note upstream also caps max_tool_calls at 100, which is its real bound. We
  // cannot mirror that: Claude Code owns the agent loop, so we have no
  // mid-run hook to count calls. The per-cell timeout is our only equivalent
  // backstop, which is why it moves with maxTurns rather than staying at 300s.
  //
  // 256 matches upstream MCP-Atlas (services/agent-harness: DEFAULT_MAX_TURNS).
  // Deliberately NOT a cost bound, despite runaway cells being the largest cost
  // tail here — one 31-turn cell was 91% of the native-vs-ratel cost delta in
  // the 5-task Sonnet 4.6 smoke.
  //
  // The reason a cap is the wrong tool for that: it does not bind symmetrically.
  // ratel spends about one extra turn per search and ran +56% turns in that
  // smoke, so any cap low enough to bound cost binds on ratel first, and a
  // truncated cell scores as a failure. That manufactures an arm difference out
  // of the harness rather than the gateway. Measured: the old hardcoded 20
  // terminated 5 of 15 ratel cells and zero native cells.
  //
  // Cost is bounded by --dollar-global and by the per-cell timeout instead, both
  // of which stop a runaway without silently converting it into a lost task.
  //
  // If a run is ever capped lower, count `finish_reason === "error_max_turns"`
  // per arm before reading any success delta: non-zero on ratel and zero on
  // native voids the comparison.
  const maxTurns = Math.max(1, Number(arg("--max-turns", "256")));
  const perCellTimeoutMs = Math.max(1000, Number(arg("--per-cell-timeout-ms", "1800000")));
  const retrieverMethodArg = arg("--retriever-method", "bm25");
  if (!["bm25", "semantic", "hybrid"].includes(retrieverMethodArg)) {
    console.error(
      `--retriever-method must be one of bm25, semantic, hybrid — got "${retrieverMethodArg}"`,
    );
    process.exitCode = 1;
    return;
  }
  const retrieverMethod = retrieverMethodArg as "bm25" | "semantic" | "hybrid";
  // Which agent CLI drives the cells. An experimental dimension, not a
  // convenience switch: harness is part of cell keys, the native cache key,
  // every summary group key, and the report nesting.
  const harnessArg = arg("--harness", "claude-code");
  if (!["claude-code", "codex"].includes(harnessArg)) {
    console.error(`--harness must be one of claude-code, codex — got "${harnessArg}"`);
    process.exitCode = 1;
    return;
  }
  const harness = harnessArg as AgentHarness;
  let codexPricing: CodexPricing | undefined;
  if (harness === "codex") {
    // No silent claude-model default on a codex run: the model names an
    // OpenAI model and doubles as the pricing-table key.
    const modelExplicit =
      process.argv.includes("--model") || Boolean(process.env.RATEL_BENCH_MODEL);
    if (!modelExplicit) {
      console.error(
        "--harness codex requires an explicit --model (e.g. gpt-5.6-luna); " +
          "the claude default does not apply",
      );
      process.exitCode = 1;
      return;
    }
    codexPricing = CODEX_PRICING[model];
    if (!codexPricing) {
      console.error(
        `no pricing entry for codex model "${model}" — add it to CODEX_PRICING in ` +
          "mcpatlas-codex.ts (cost enforcement for --dollar-global depends on it)",
      );
      process.exitCode = 1;
      return;
    }
  }

  const manifest = JSON.parse(
    readFileSync(resolveRepoPath(`fixtures/mcpatlas/catalog-${scope}.json`), "utf8"),
  ) as McpAtlasCatalogManifest;
  const pinned = JSON.parse(
    readFileSync(resolveRepoPath("fixtures/mcpatlas/tasks-coding-v1.json"), "utf8"),
  ) as { task_list_hash: string; dataset_revision?: string };
  // Not fatal — corpus CONTENT is guarded by task_list_hash below — but a
  // moving revision means nobody can later recover which upstream state this
  // corpus came from, so say it loudly rather than record it silently.
  if (!isImmutableRevision(pinned.dataset_revision ?? "unpinned")) {
    console.warn(
      `warning: dataset_revision "${pinned.dataset_revision ?? "unpinned"}" is not an ` +
        "immutable commit reference — provenance for this run cannot be reproduced from it. " +
        "Re-run mcpatlas-ingest with a pinned --dataset-revision.",
    );
  }
  const allTasks = readJsonl<McpAtlasTask>(resolveRepoPath("test-data/mcpatlas-coding.jsonl"));
  // Offset first, then limit — so `--tasks 1 --task-offset 3` selects exactly
  // the 4th task rather than requiring a re-run of every task before it just
  // to isolate one for debugging (a real friction point today: `--tasks N`
  // alone always starts from the first task).
  const offsetTasks = tasksOffset > 0 ? allTasks.slice(tasksOffset) : allTasks;
  const tasks = tasksLimit > 0 ? offsetTasks.slice(0, tasksLimit) : offsetTasks;

  if (!skipDoctor) {
    const { results, facts } = await runChecks({
      scope,
      manifest,
      taskCount: allTasks.length,
      taskListHash: pinned.task_list_hash,
      expectedTaskListHash: pinned.task_list_hash,
      sandboxUrl,
      ratelLocalPin,
      probes: defaultProbes(),
      harness,
    });
    if (doctorExitCode(results) !== 0) {
      console.error(formatResults(results));
      process.exitCode = 1;
      return;
    }
    console.log(formatResults(results));
    await runMain(facts);
  } else {
    await runMain({
      claude_code_version: "unknown",
      codex_version: harness === "codex" ? "unknown" : null,
      ratel_local_version: ratelLocalPin,
      ratel_sdk_version: null,
      sandbox_url: sandboxUrl,
      sandbox_tool_count: null,
      catalog_sha256: manifest.catalog_sha256,
    });
  }

  async function runMain(facts: ResolvedFactsShape): Promise<void> {
    let benchGitSha = "unknown";
    try {
      benchGitSha = sh("git", ["rev-parse", "HEAD"]);
    } catch {
      // best-effort
    }

    const cfgCore = buildRunConfig({
      scope,
      manifest,
      ratelVersionLabel: ratelLocalPin,
      ratelLocalVersion: facts.ratel_local_version ?? ratelLocalPin,
      ratelSdkVersion: facts.ratel_sdk_version,
      claudeCodeVersion: facts.claude_code_version ?? "unknown",
      benchGitSha,
      agentModel: model,
      maxTurns,
      perCellTimeoutMs,
      permissionMode: "bypassPermissions",
      judgeModel: judgeModelId,
      retrieverMethod,
      topKTools: 5,
      topKSkills: 3,
      arms,
      evalKs: [1, 3, 5],
      catalogTools,
      runsPerTask,
      seed: 0,
      concurrency,
      datasetRevision: pinned.dataset_revision ?? "unpinned",
      taskListHash: pinned.task_list_hash,
      taskIds: tasks.map((t) => t.task_id),
      sandboxUrl,
      atlasImageDigests: {},
      dollarCapGlobal: dollarCap,
      declaredLimitations: [],
      harness,
      ...(harness === "codex"
        ? {
            codexVersion: facts.codex_version ?? "unknown",
            codexPricing,
            codexLockdown: codexLockdown(),
          }
        : {}),
    });

    const priorOutput = readJsonl<McpAtlasCell>(outputPath);
    if (priorOutput.length && !force) {
      const priorHash = priorOutput[0]?.config_hash;
      const priorTaskListHash = readJsonl<McpAtlasRunConfig>(
        resolveRepoPath("results/raw/mcpatlas/config-history.jsonl"),
      ).find((c) => c.config_hash === priorHash)?.corpus.task_list_hash;
      if (priorTaskListHash && priorTaskListHash !== pinned.task_list_hash) {
        console.error(
          `refusing to run: existing ${outputPath} was built from a different task list ` +
            `(${priorTaskListHash} vs ${pinned.task_list_hash}). Pass --force to override.`,
        );
        process.exitCode = 1;
        return;
      }
    }

    const runId = `mcpatlas-${scope}-${Date.now()}`;
    const generatedAt = new Date().toISOString();
    const cfg = freezeConfig(cfgCore, runId, generatedAt);
    const configDir = resolveRepoPath("results/raw/mcpatlas/config");
    mkdirSync(configDir, { recursive: true });
    writeFileSync(join(configDir, `${runId}.json`), JSON.stringify(cfg, null, 2));
    appendJsonl(resolveRepoPath("results/raw/mcpatlas/config-history.jsonl"), cfg);

    const missing = missingEnv(manifest, process.env);
    if (missing.length) {
      console.error(`missing required credentials: ${missing.join(", ")}`);
      process.exitCode = 1;
      return;
    }

    const shim: ShimSpec = {
      shimPath: resolveRepoPath("agent-eval/src/atlas-mcp-shim.ts"),
      sandboxUrl,
    };
    const handle = ensureSandbox({
      containerName: "mcpatlas-sandbox",
      image: "ghcr.io/scaleapi/mcp-atlas:1.2.7",
      port: 1984,
      envFile: resolveRepoPath("fixtures/mcpatlas/upstream/.env"),
      templateMount: {
        hostPath: resolveRepoPath("agent-eval/sandbox/mcp_server_template.json"),
        containerPath: "/agent-environment/src/agent_environment/mcp_server_template.json",
      },
    });
    const healthy = await waitForHealth(sandboxUrl, 300_000);
    if (!healthy) {
      console.error(`sandbox at ${sandboxUrl} never became healthy`);
      teardownSandbox(handle, "mcpatlas-sandbox", keepStack);
      process.exitCode = 1;
      return;
    }

    let judgeModel: LanguageModel | undefined;
    if (judgeModelId) {
      if (!process.env.ANTHROPIC_API_KEY) {
        console.error("--judge-model was given but ANTHROPIC_API_KEY is not set");
        teardownSandbox(handle, "mcpatlas-sandbox", keepStack);
        process.exitCode = 1;
        return;
      }
      const { createAnthropic } = await import("@ai-sdk/anthropic");
      judgeModel = createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY })(judgeModelId);
    }

    const queue = buildQueue(tasks, arms, cfg.runs_per_task);

    let cacheSourceCells: McpAtlasCell[] = [];
    if (cacheSourcePath)
      cacheSourceCells = readJsonl<McpAtlasCell>(resolveRepoPath(cacheSourcePath));
    const cacheIndex = readNativeCacheIndex(cacheSourceCells, {
      promptHash: PROMPT_HASH,
      taskListHash: pinned.task_list_hash,
      datasetRevision: cfg.corpus.dataset_revision,
    });
    const { toRun, reusedCells } = drainNativeCache(
      queue,
      (item) =>
        nativeCacheKey({
          taskId: item.task.task_id,
          model,
          scope,
          catalogTools: cfg.catalog_tools,
          runIndex: item.runIndex,
          agentVersion:
            cfg.agent_harness === "codex"
              ? (cfg.codex_version ?? "unknown")
              : cfg.claude_code_version,
          promptHash: PROMPT_HASH,
          taskListHash: pinned.task_list_hash,
          datasetRevision: cfg.corpus.dataset_revision,
          harness: cfg.agent_harness,
        }),
      cacheIndex.reuse,
      refreshNative,
      (cell, item) => ({
        ...cell,
        run_id: cfg.run_id,
        config_hash: cfg.config_hash,
        generated_at: cfg.generated_at,
        cell_key: cellKeyFor(item, scope, cfg.agent_harness),
        cache_source: "reused",
      }),
    );

    truncateJsonl(outputPath);
    const toolCallsPath = resolveRepoPath("results/raw/mcpatlas/tool-calls.jsonl");
    const searchEventsPath = resolveRepoPath("results/raw/mcpatlas/search-events.jsonl");
    const retrievalRowsPath = resolveRepoPath("results/raw/mcpatlas/retrieval-rows.jsonl");
    truncateJsonl(toolCallsPath);
    truncateJsonl(searchEventsPath);
    truncateJsonl(retrievalRowsPath);

    for (const c of reusedCells) appendJsonl(outputPath, c);

    const scratchRoot = resolveRepoPath("results/raw/mcpatlas/cells");

    // Pass 1: native cells, so their tool-call rows establish the baseline
    // before any ratel cell needs `nativeBaselineMs`/`nativeCatalogTokens`.
    const nativeItems = toRun.filter((q) => q.arm === "native");
    const ratelItems = toRun.filter((q) => q.arm === "ratel");
    let nativeBaselineMs = collectNativeBaselineMs([]);
    const collectedNativeToolCalls: McpAtlasToolCallRow[] = [];

    // Measured once, before any cell, and fixed for the campaign: the catalog
    // is registered whole and identical across arms, so these do not vary per
    // task. Both come from `schemaTokenEstimate` — see its docstring on why one
    // ruler for both arms matters more than absolute precision.
    const sandboxTools = await fetchSandboxTools(sandboxUrl);
    const nativeCatalogTokens = sandboxTools ? schemaTokenEstimate(sandboxTools) : 0;
    const knownServersForTokens = manifest.servers.map((sv) => sv.server);
    const perToolTokens = sandboxTools
      ? perToolTokenMap(
          sandboxTools,
          (t) => normalizeToolId(t.name, knownServersForTokens) ?? t.name,
        )
      : undefined;
    const gatewaySchemaTokens =
      (await measureGatewaySchemaTokens({
        manifest,
        shim,
        retrieverMethod: cfg.retriever_method,
        ratelLocalPin: cfg.ratel_local_version,
        scratchRoot,
      })) ?? 0;
    if (!nativeCatalogTokens || !gatewaySchemaTokens) {
      // Not fatal — the run is still valid for task success, selection and
      // cost. Say so loudly, because a silent zero here is exactly the bug
      // this measurement replaced.
      console.warn(
        `warning: schema occupancy unmeasured (catalog=${nativeCatalogTokens}, gateway=${gatewaySchemaTokens}); ` +
          "tool_schema_tokens and the savings derived from it will be 0 for this run",
      );
    } else {
      console.log(
        `schema occupancy: native catalog ${nativeCatalogTokens} tokens (${sandboxTools?.length ?? 0} tools), ` +
          `gateway ${gatewaySchemaTokens} tokens`,
      );
    }

    const runOneNative = (item: QueueItem) =>
      runCell({
        item,
        cfg,
        manifest,
        shim,
        scratchRoot,
        keepArtifacts,
        judgeModel,
        nativeBaselineMs: new Map(),
        nativeCatalogTokens,
        gatewaySchemaTokens,
        perToolTokens,
        cacheSource: "live",
      });

    const nativeSummary = await runCampaign(nativeItems, {
      concurrency,
      dollarCap,
      runOne: runOneNative,
      onCell: (r) => {
        appendJsonl(outputPath, r.cell);
        for (const row of r.toolCallRows) {
          appendJsonl(toolCallsPath, row);
          collectedNativeToolCalls.push(row);
        }
      },
    });

    nativeBaselineMs = collectNativeBaselineMs(collectedNativeToolCalls);

    const runOneRatel = (item: QueueItem) =>
      runCell({
        item,
        cfg,
        manifest,
        shim,
        scratchRoot,
        keepArtifacts,
        judgeModel,
        nativeBaselineMs,
        nativeCatalogTokens,
        gatewaySchemaTokens,
        perToolTokens,
        cacheSource: "live",
      });

    const ratelSummary = await runCampaign(ratelItems, {
      concurrency,
      dollarCap: dollarCap != null ? Math.max(0, dollarCap - nativeSummary.total_dollars) : null,
      runOne: runOneRatel,
      onCell: (r) => {
        appendJsonl(outputPath, r.cell);
        for (const row of r.toolCallRows) appendJsonl(toolCallsPath, row);
        for (const row of r.searchEventRows) appendJsonl(searchEventsPath, row);
        for (const row of r.retrievalRows) appendJsonl(retrievalRowsPath, row);
      },
    });

    teardownSandbox(handle, "mcpatlas-sandbox", keepStack);

    const summary: CampaignSummary = {
      cells_run: nativeSummary.cells_run + ratelSummary.cells_run,
      cells_skipped: nativeSummary.cells_skipped + ratelSummary.cells_skipped,
      total_dollars: nativeSummary.total_dollars + ratelSummary.total_dollars,
      stopped_reason:
        nativeSummary.stopped_reason === "global_cap" ||
        ratelSummary.stopped_reason === "global_cap"
          ? "global_cap"
          : "completed",
    };
    console.log(formatDoneLine(summary, reusedCells.length));
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main()
    .catch((err) => {
      console.error(`mcpatlas-run: ${(err as Error).message}`);
      process.exitCode = 1;
    })
    .finally(() => {
      // Guarantee termination once the work is done and the `done:` line is out.
      //
      // Everything above is finished at this point: rows are written with
      // synchronous writeFileSync, so there is no pending output to lose. What
      // remains is whether the event loop DRAINS, and this driver spawns Docker,
      // npx, Claude Code and eleven stdio shims per cell — any one leaked child
      // or undestroyed pipe holds a handle open and the process simply never
      // returns. That already happened once: the schema probe killed `npx`
      // rather than its process group, and a completed 2-cell run hung after
      // printing its final line.
      //
      // On a laptop that is an annoyance you can Ctrl-C. On CodeBuild it is a
      // 45-minute timeout, a failed build, and no signal saying the benchmark
      // had actually finished — the run looks broken when the data is fine.
      //
      // unref'd deliberately: an unref'd timer cannot hold the loop open by
      // itself, so a clean exit still happens immediately and this never fires.
      // It only fires when something ELSE is keeping the loop alive, which is
      // precisely the case worth killing. The delay lets a piped stdout flush,
      // since writes to a pipe are asynchronous and process.exit() would
      // truncate the `done:` line that downstream tooling parses.
      setTimeout(() => process.exit(process.exitCode ?? 0), 2_000).unref();
    });
}
