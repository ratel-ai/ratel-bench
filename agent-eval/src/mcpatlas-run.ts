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
import {
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
  parseUses,
} from "./mcpatlas-build.js";
import { invokeSpans, parseTelemetry } from "./mcpatlas-gateway.js";
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
  type ShimSpec,
} from "./mcpatlas-servers.js";
import type {
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
}

export function buildRunConfig(input: BuildRunConfigInput): FrozenConfigCore {
  return {
    run_type: "mcpatlas_config",
    ratel_version_label: input.ratelVersionLabel,
    ratel_local_version: input.ratelLocalVersion,
    ratel_sdk_version: input.ratelSdkVersion,
    claude_code_version: input.claudeCodeVersion,
    bench_git_sha: input.benchGitSha,
    agent_harness: "claude-code",
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
  runIndex: number;
  agentVersion: string;
  promptHash: string;
  taskListHash: string;
  datasetRevision: string;
}

/**
 * The native arm never touches ratel-local, so it can be reused across
 * `ratel_local_version` changes — deliberately absent from this key. `scope`
 * IS included: here the catalog size is the tool surface itself, and omitting
 * it would serve a 79-tool cell for a 195-tool cell.
 */
export function nativeCacheKey(input: NativeCacheKeyInput): string {
  return [
    input.taskId,
    "native",
    input.model,
    input.scope,
    input.runIndex,
    input.agentVersion,
    input.promptHash,
    input.taskListHash,
    input.datasetRevision,
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
      runIndex: c.run_index,
      agentVersion: c.agent_version,
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
  judgeClaims: typeof judgeClaims;
}

export const REAL_RUN_CELL_DEPS: RunCellDeps = { runClaude: realRunClaude, judgeClaims };

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

function cellKeyFor(item: QueueItem, scope: McpAtlasScope): string {
  return `${item.task.task_id}__${item.arm}__s${scope}__r${item.runIndex}`;
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

/** Runs one (task, arm) cell end to end: writes the cell's mcp.json/ratel.json,
 *  invokes Claude Code, reads the transcript and telemetry, judges the claims,
 *  and assembles all four row types. On any failure — including the case that
 *  matters most, an empty telemetry file on a `ratel` cell — returns a
 *  complete, schema-valid result with `cell.error` set, never a missing row. */
export async function runCell(o: RunCellOptions): Promise<RunCellResult> {
  const deps = o.deps ?? REAL_RUN_CELL_DEPS;
  const { item, cfg, manifest } = o;
  const cellKey = cellKeyFor(item, cfg.catalogs[0].scope);
  const scratch = makeScratch(cellKey, o.scratchRoot);
  const knownServers = manifest.servers.map((s) => s.server);

  try {
    if (item.arm === "native") {
      writeFileSync(scratch.mcpConfigPath, JSON.stringify(buildNativeMcpConfig(manifest, o.shim)));
    } else {
      writeFileSync(
        scratch.serveConfigPath,
        JSON.stringify(buildRatelServeConfig(manifest, o.shim, cfg.retriever_method)),
      );
      writeFileSync(
        scratch.mcpConfigPath,
        JSON.stringify(
          buildRatelMcpConfig({
            ratelLocalPin: cfg.ratel_local_version,
            serveConfigPath: scratch.serveConfigPath,
            telemetryPath: scratch.telemetryPath,
          }),
        ),
      );
    }

    const allowedTools =
      item.arm === "native"
        ? manifest.servers.flatMap((s) =>
            s.tool_ids.map((id) => `mcp__${id.split("/")[0]}__${id.split("/").slice(1).join("/")}`),
          )
        : GATEWAY_TOOLS.map((t) => `mcp__ratel-local__${t}`);

    const outcome = await deps.runClaude({
      prompt: buildPrompt(item.task.prompt),
      mcpConfigPath: scratch.mcpConfigPath,
      allowedTools,
      model: cfg.agent_model,
      maxTurns: cfg.max_turns,
      permissionMode: cfg.permission_mode,
      cwd: scratch.workspaceDir,
      homeDir: scratch.homeDir,
      env: inheritedEnv(),
      timeoutMs: cfg.per_cell_timeout_ms,
      appendSystemPrompt: SYSTEM_PROMPT_ADDENDUM,
    });

    const result = parseClaudeResult(outcome.stdout);
    if (!result) {
      throw new Error(
        `claude produced no parseable result envelope (timedOut=${outcome.timedOut}, exitCode=${outcome.exitCode}): ${outcome.stderr.slice(0, 500)}`,
      );
    }

    const tPath = transcriptPath(scratch.homeDir, scratch.workspaceDir, result.session_id);
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

    const ctx: CellContext = {
      run_id: cfg.run_id,
      config_hash: cfg.config_hash,
      generated_at: cfg.generated_at,
      cell_key: cellKey,
      task: item.task,
      arm: item.arm,
      catalog_scope: cfg.catalogs[0].scope,
      catalog_tool_ids: manifest.servers.flatMap((s) => s.tool_ids),
      eval_ks: cfg.eval_ks,
      per_call_timeout_ms: cfg.per_cell_timeout_ms,
      model: cfg.agent_model,
      ratel_version_label: cfg.ratel_version_label,
      ratel_local_version: cfg.ratel_local_version,
      ratel_sdk_version: cfg.ratel_sdk_version,
    };

    const claimRubric = await deps.judgeClaims({
      taskId: item.task.task_id,
      prompt: item.task.prompt,
      claims: item.task.claims,
      finalText: result.result,
      model: o.judgeModel,
    });

    const cell = assembleCell({
      ctx,
      result,
      transcriptText,
      transcriptPath: tPath ?? "",
      telemetryText,
      telemetryPath: item.arm === "ratel" ? scratch.telemetryPath : null,
      claimRubric,
      nativeCatalogTokens: o.nativeCatalogTokens,
      gatewaySchemaTokens: o.gatewaySchemaTokens,
      nativeBaselineMs: o.nativeBaselineMs,
      agentVersion: cfg.claude_code_version,
      runIndex: item.runIndex,
      cacheSource: o.cacheSource,
    });

    const uses = parseUses(transcriptText);
    const { calls, offCatalog } = effectiveCalls(uses, knownServers);
    const spans = invokeSpans(parseTelemetry(telemetryText), knownServers);
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
      catalog_size: item.arm === "native" ? manifest.tool_count : GATEWAY_TOOLS.length,
      run_index: item.runIndex,
      ratel_version_label: cfg.ratel_version_label,
      ratel_local_version: cfg.ratel_local_version,
      ratel_sdk_version: cfg.ratel_sdk_version,
      agent_version: cfg.claude_code_version,
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
  const retrieverMethodArg = arg("--retriever-method", "bm25");
  if (!["bm25", "semantic", "hybrid"].includes(retrieverMethodArg)) {
    console.error(
      `--retriever-method must be one of bm25, semantic, hybrid — got "${retrieverMethodArg}"`,
    );
    process.exitCode = 1;
    return;
  }
  const retrieverMethod = retrieverMethodArg as "bm25" | "semantic" | "hybrid";

  const manifest = JSON.parse(
    readFileSync(resolveRepoPath(`fixtures/mcpatlas/catalog-${scope}.json`), "utf8"),
  ) as McpAtlasCatalogManifest;
  const pinned = JSON.parse(
    readFileSync(resolveRepoPath("fixtures/mcpatlas/tasks-coding-v1.json"), "utf8"),
  ) as { task_list_hash: string; dataset_revision?: string };
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
      maxTurns: 20,
      perCellTimeoutMs: 300_000,
      permissionMode: "bypassPermissions",
      judgeModel: judgeModelId,
      retrieverMethod,
      topKTools: 5,
      topKSkills: 3,
      arms,
      evalKs: [1, 3, 5],
      runsPerTask: 1,
      seed: 0,
      concurrency,
      datasetRevision: pinned.dataset_revision ?? "unpinned",
      taskListHash: pinned.task_list_hash,
      taskIds: tasks.map((t) => t.task_id),
      sandboxUrl,
      atlasImageDigests: {},
      dollarCapGlobal: dollarCap,
      declaredLimitations: [],
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
          runIndex: item.runIndex,
          agentVersion: cfg.claude_code_version,
          promptHash: PROMPT_HASH,
          taskListHash: pinned.task_list_hash,
          datasetRevision: cfg.corpus.dataset_revision,
        }),
      cacheIndex.reuse,
      refreshNative,
      (cell, item) => ({
        ...cell,
        run_id: cfg.run_id,
        config_hash: cfg.config_hash,
        generated_at: cfg.generated_at,
        cell_key: cellKeyFor(item, scope),
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
    let nativeCatalogTokens = 0;
    let gatewaySchemaTokens = 0;
    const collectedNativeToolCalls: McpAtlasToolCallRow[] = [];

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
        nativeCatalogTokens: 0,
        gatewaySchemaTokens: 0,
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
        if (r.cell.tokens.tool_schema_tokens > 0) {
          nativeCatalogTokens = Math.max(nativeCatalogTokens, r.cell.tokens.tool_schema_tokens);
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
        if (r.cell.tokens.tool_schema_tokens > 0) {
          gatewaySchemaTokens = Math.max(gatewaySchemaTokens, r.cell.tokens.tool_schema_tokens);
        }
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
  main().catch((err) => {
    console.error(`mcpatlas-run: ${(err as Error).message}`);
    process.exit(1);
  });
}
