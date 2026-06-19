// Unified entrypoint for the benchmark suite. Drives both retrieval modes
// (MetaTool + ToolRet, the Rust crate), then the agent campaign (mode (c)) if
// at least one provider key is set, then emits the merged REPORT.md.
//
// Behavior:
//   1. Ingest each corpus if its normalized JSONL is missing (`--download`).
//   2. Run BM25 retrieval over each corpus at corpus-appropriate pool sizes.
//   3. If `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set (and `--skip-agent` is
//      not), spawn the agent campaign with conservative defaults — small
//      sampled subset, 1 run/cell, all three arms, $5 global cap. Skipped with
//      a notice otherwise so the rest of run-all stays $0 and CI-friendly.
//   4. Render REPORT.md from the retrieval JSONLs (and agent.jsonl when present).
//
// Flags:
//   --force         re-ingest even if the snapshot already exists
//   --skip-ingest   never call the ingest CLI (fail loudly if missing)
//   --skip-agent    never run mode (c), even if a provider key is present
//   --only NAME     restrict to a single corpus: "metatool" | "toolret" | "sragents"
//   --bfcl          run ONLY the self-contained BFCL pipeline (ingest both
//                   subsets → BM25 retrieval → qwen3.5 campaign → BFCL.json →
//                   delete downloaded data). Requires a local Ollama with
//                   qwen3.5 unless paired with --skip-agent.
//   --keep-bfcl     with --bfcl, skip the post-run cleanup (keep fixtures +
//                   normalized corpora for debugging)

import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, rmSync, statSync } from "node:fs";
import { REPO_ROOT, resolveRepoPath } from "./paths.js";

type CorpusName = "metatool" | "toolret" | "bfcl-simple" | "bfcl-multiple";
type TargetName = "metatool" | "toolret" | "sragents";

interface CorpusSpec {
  name: CorpusName;
  corpus: string;
  retrievalOut: string;
  poolSizes: string;
  topK: string;
}

const CORPORA: CorpusSpec[] = [
  {
    name: "metatool",
    corpus: "test-data/metatool.jsonl",
    retrievalOut: "results/metatool-retrieval.jsonl",
    // MetaTool's gold-tool universe ceiling is ~199 plugins; pool sizes stay at
    // or below it so every cell is meaningful.
    poolSizes: "30,100,180",
    topK: "1,3,5,10",
  },
  {
    name: "toolret",
    corpus: "test-data/toolret.jsonl",
    retrievalOut: "results/toolret-retrieval.jsonl",
    // ToolRet's gold-only universe is ~7,651 unique tools — small / mid / full.
    poolSizes: "100,1000,7000",
    topK: "1,3,5,10",
  },
];

/**
 * BFCL is a focused, self-contained pipeline (own ingest that emits two files,
 * a qwen3.5 agent campaign, a dedicated report, and a cleanup of downloaded
 * data). It is opt-in via `--bfcl` so the default run-all stays $0/key-free and
 * doesn't require a local Ollama. Retrieval pool sizes go up to each subset's
 * function-universe ceiling (~400 simple / ~600 multiple).
 */
const BFCL_CORPORA: CorpusSpec[] = [
  {
    name: "bfcl-simple",
    corpus: "test-data/bfcl-simple.jsonl",
    retrievalOut: "results/bfcl-simple-retrieval.jsonl",
    poolSizes: "30,100,400",
    topK: "1,3,5,10",
  },
  {
    name: "bfcl-multiple",
    corpus: "test-data/bfcl-multiple.jsonl",
    retrievalOut: "results/bfcl-multiple-retrieval.jsonl",
    poolSizes: "30,100,600",
    topK: "1,3,5,10",
  },
];

/**
 * Pool size the BFCL **agent** campaign runs at (distinct from the retrieval
 * sweep above). control-baseline puts the whole pool in context, so this is the
 * "fat context" the ratel arms are saving against — 100 distractors is enough to
 * make the savings meaningful while still fitting a local qwen3.5 context and
 * keeping a full-dataset run tractable.
 */
const BFCL_AGENT_POOL_SIZE = "100";

/**
 * Model the BFCL agent campaign runs on (local Ollama, $0). The `ollama:` prefix
 * routes through the local server; the tag must exist there (`ollama list`).
 */
const BFCL_AGENT_MODEL = "ollama:qwen3.5:4b";

// SR-Agents skill corpus (separate from the tool corpora above): an authored
// skill catalog (~26k skills) is the BM25 index, and instances carry gold skill
// ids. Runs via the dedicated `skill-retrieval` subcommand.
const SRAGENTS = {
  catalog: "test-data/sragents-skills.jsonl",
  instances: "test-data/sragents.jsonl",
  retrievalOut: "results/sragents-skill-retrieval.jsonl",
  // Catalog is ~26k skills — small / mid / full.
  poolSizes: "100,1000,26262",
  topK: "1,3,5,10",
} as const;

function hasFlag(name: string): boolean {
  return process.argv.includes(name);
}

function flagValue(name: string): string | undefined {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : undefined;
}

function runStep(label: string, bin: string, args: string[]): void {
  console.log(`\n→ ${label}`);
  console.log(`  $ ${bin} ${args.join(" ")}`);
  const res = spawnSync(bin, args, { stdio: "inherit", cwd: REPO_ROOT });
  if (res.status !== 0) {
    throw new Error(`${label} failed (exit ${res.status ?? "?"})`);
  }
}

function isNonEmptyDir(p: string): boolean {
  try {
    return statSync(p).isDirectory() && readdirSync(p).length > 0;
  } catch {
    return false;
  }
}

function ingest(spec: CorpusSpec, force: boolean, skipIngest: boolean): void {
  const absCorpus = resolveRepoPath(spec.corpus);
  if (existsSync(absCorpus) && !force) {
    console.log(`✓ ${spec.name}: corpus present at ${spec.corpus}, skipping ingest`);
    return;
  }
  if (skipIngest) {
    throw new Error(
      `${spec.name}: ${spec.corpus} missing and --skip-ingest set. ` +
        `Run \`cargo run -p ratel-benchmark-retrieval --release -- ingest ${spec.name} --download\` first.`,
    );
  }
  // Re-use cached upstream fixtures when present — only call `--download` on a
  // truly clean clone or with `--force`.
  const fixturesDir = resolveRepoPath(`fixtures/${spec.name}`);
  const args = ["run", "-p", "ratel-benchmark-retrieval", "--release", "--", "ingest", spec.name];
  if (force || !isNonEmptyDir(fixturesDir)) {
    args.push("--download");
  } else {
    console.log(`  (fixtures cached at ${fixturesDir} — skipping --download)`);
  }
  runStep(`ingest ${spec.name}`, "cargo", args);
}

function retrieval(spec: CorpusSpec): void {
  runStep(`retrieval ${spec.name}`, "cargo", [
    "run",
    "-p",
    "ratel-benchmark-retrieval",
    "--release",
    "--",
    "retrieval",
    "--corpus",
    spec.corpus,
    "--output",
    spec.retrievalOut,
    "--top-k",
    spec.topK,
    "--pool-sizes",
    spec.poolSizes,
  ]);
}

function ingestSragents(force: boolean, skipIngest: boolean): void {
  const haveCatalog = existsSync(resolveRepoPath(SRAGENTS.catalog));
  const haveInstances = existsSync(resolveRepoPath(SRAGENTS.instances));
  if (haveCatalog && haveInstances && !force) {
    console.log(`✓ sragents: catalog + instances present, skipping ingest`);
    return;
  }
  if (skipIngest) {
    throw new Error(
      `sragents: ${SRAGENTS.catalog} / ${SRAGENTS.instances} missing and --skip-ingest set. ` +
        `Run \`cargo run -p ratel-benchmark-retrieval --release -- ingest sragents --download\` first.`,
    );
  }
  const fixturesDir = resolveRepoPath("fixtures/sragents");
  const args = ["run", "-p", "ratel-benchmark-retrieval", "--release", "--", "ingest", "sragents"];
  if (force || !isNonEmptyDir(fixturesDir)) {
    args.push("--download");
  } else {
    console.log(`  (fixtures cached at ${fixturesDir} — skipping --download)`);
  }
  runStep("ingest sragents", "cargo", args);
}

function skillRetrieval(): void {
  runStep("skill-retrieval sragents", "cargo", [
    "run",
    "-p",
    "ratel-benchmark-retrieval",
    "--release",
    "--",
    "skill-retrieval",
    "--instances",
    SRAGENTS.instances,
    "--skills-catalog",
    SRAGENTS.catalog,
    "--output",
    SRAGENTS.retrievalOut,
    "--top-k",
    SRAGENTS.topK,
    "--pool-sizes",
    SRAGENTS.poolSizes,
  ]);
}

/**
 * Mode (c) — agent campaign. Gated on a provider key being available, so a
 * clean clone with no `.env` still runs the free retrieval modes + report.
 *
 * The `--skip-agent` flag short-circuits this even when keys are present (e.g.
 * for a fast iteration on the report layout). Defaults are deliberately
 * conservative — a developer iterating locally shouldn't blow $25 by typing
 * `pnpm run-all`. The first headline campaign is launched explicitly via
 * `pnpm -F @ratel-ai/benchmark start --runs 5 ...` (see agent/README.md).
 */
function agentCampaign(skipAgent: boolean): void {
  if (skipAgent) {
    console.log("\n→ mode (c) — agent campaign\n  --skip-agent set, skipping.");
    return;
  }
  const hasOpenAI = !!process.env.OPENAI_API_KEY;
  const hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
  if (!hasOpenAI && !hasAnthropic) {
    console.log(
      "\n→ mode (c) — agent campaign\n" +
        "  no provider key set (OPENAI_API_KEY / ANTHROPIC_API_KEY). Skipping. " +
        "Set one and re-run, or invoke `pnpm -F @ratel-ai/benchmark start ...` directly.",
    );
    return;
  }
  const models: string[] = [];
  if (hasAnthropic) models.push("claude-sonnet-4-6");
  if (hasOpenAI) models.push("gpt-5.4-mini");

  // Conservative defaults for an automated invocation: small sampled subset,
  // 1 run per cell, every committed arm (control + 3 ratel ablations), $5 cap.
  // The local-only `claude-sdk-tool-search` arm is excluded by default — opt
  // into it via `pnpm -F @ratel-ai/benchmark start --arms ...,claude-sdk-tool-search`.
  // Override these defaults by calling `pnpm -F @ratel-ai/benchmark start` directly.
  runStep("agent campaign (mode c)", "pnpm", [
    "-F",
    "@ratel-ai/benchmark",
    "start",
    "--scenarios",
    "50",
    "--runs",
    "1",
    "--arms",
    "control-baseline,control-oracle,ratel-full,ratel-pre-discovery,ratel-discovery-tool",
    "--models",
    models.join(","),
    "--pool-sizes",
    "180",
    "--dollar-global",
    "5",
    "--concurrency",
    "10",
    "--quiet",
  ]);
}

function report(): void {
  // Re-uses the existing report CLI; auto-discovers `*retrieval.jsonl` under
  // `results/` and joins with `agent.jsonl` if present.
  runStep("render REPORT.md", "pnpm", ["-F", "@ratel-ai/benchmark", "report"]);
}

/** One cargo call emits BOTH normalized BFCL corpora (simple + multiple). */
function ingestBfcl(force: boolean, skipIngest: boolean): void {
  const simple = resolveRepoPath("test-data/bfcl-simple.jsonl");
  const multiple = resolveRepoPath("test-data/bfcl-multiple.jsonl");
  if (existsSync(simple) && existsSync(multiple) && !force) {
    console.log("✓ bfcl: corpora present, skipping ingest");
    return;
  }
  if (skipIngest) {
    throw new Error(
      "bfcl: corpora missing and --skip-ingest set. Run " +
        "`cargo run -p ratel-benchmark-retrieval --release -- ingest bfcl --download` first.",
    );
  }
  const fixturesDir = resolveRepoPath("fixtures/bfcl");
  const args = ["run", "-p", "ratel-benchmark-retrieval", "--release", "--", "ingest", "bfcl"];
  if (force || !isNonEmptyDir(fixturesDir)) {
    args.push("--download");
  } else {
    console.log(`  (fixtures cached at ${fixturesDir} — skipping --download)`);
  }
  runStep("ingest bfcl", "cargo", args);
}

/**
 * BFCL agent campaign on qwen3.5 (local Ollama, $0). Runs both subsets with
 * control-baseline (without Ratel), ratel-full (with Ratel), and control-oracle
 * (upper bound + savings-table oracle column). Selection-only judging, so the
 * LLM judge is skipped (`--no-judge`) — no provider key needed. Full datasets:
 * no `--scenarios` cap.
 */
function bfclAgentCampaign(skipAgent: boolean): void {
  if (skipAgent) {
    console.log("\n→ BFCL agent campaign\n  --skip-agent set, skipping.");
    return;
  }
  for (const spec of BFCL_CORPORA) {
    runStep(`BFCL agent campaign ${spec.name} (${BFCL_AGENT_MODEL})`, "pnpm", [
      "-F",
      "@ratel-ai/benchmark",
      "start",
      "--corpus",
      spec.corpus,
      "--arms",
      "control-baseline,control-oracle,ratel-full",
      "--models",
      BFCL_AGENT_MODEL,
      "--pool-sizes",
      BFCL_AGENT_POOL_SIZE,
      "--runs",
      "1",
      "--no-judge",
      "--concurrency",
      "4",
      "--quiet",
    ]);
  }
}

/**
 * One consolidated `results/BFCL.json`: retrieval evaluation (split single/multi),
 * task-completion evaluation (combined, with vs without Ratel), timestamp, and
 * the ratel-ai-core / SDK versions. Replaces the per-scenario markdown.
 */
function bfclReport(): void {
  runStep("export BFCL.json", "pnpm", [
    "-F",
    "@ratel-ai/benchmark",
    "exec",
    "tsx",
    "src/bfcl-export.ts",
    "--out",
    "results/BFCL.json",
  ]);
}

/**
 * Delete downloaded/cached BFCL data after the run (per requirement). Results
 * (`results/bfcl-*-retrieval.jsonl`, `agent/results/agent.jsonl`, `results/BFCL.json`)
 * are kept — only the fixtures and the regenerable normalized corpora go.
 */
function cleanupBfcl(): void {
  for (const p of [
    "fixtures/bfcl",
    "test-data/bfcl-simple.jsonl",
    "test-data/bfcl-multiple.jsonl",
  ]) {
    rmSync(resolveRepoPath(p), { recursive: true, force: true });
  }
  console.log("✓ cleaned up BFCL fixtures + normalized corpora (results kept)");
}

/** Self-contained BFCL pipeline: ingest → retrieval → qwen campaign → report → cleanup. */
function runBfcl(force: boolean, skipIngest: boolean, skipAgent: boolean, keepData: boolean): void {
  ingestBfcl(force, skipIngest);
  for (const spec of BFCL_CORPORA) {
    retrieval(spec);
  }
  bfclAgentCampaign(skipAgent);
  bfclReport();
  if (keepData) {
    console.log("\n(--keep-bfcl set — leaving fixtures/bfcl + test-data/bfcl-*.jsonl in place)");
  } else {
    cleanupBfcl();
  }
}

function main(): void {
  const force = hasFlag("--force");
  const skipIngest = hasFlag("--skip-ingest");
  const skipAgent = hasFlag("--skip-agent");

  // BFCL is a focused, self-contained run (own report + cleanup); `--bfcl` runs
  // just that pipeline and returns.
  if (hasFlag("--bfcl")) {
    runBfcl(force, skipIngest, skipAgent, hasFlag("--keep-bfcl"));
    console.log("\n✓ BFCL run-all complete.");
    return;
  }

  const only = flagValue("--only") as TargetName | undefined;

  const validTargets: TargetName[] = ["metatool", "toolret", "sragents"];
  if (only && !validTargets.includes(only)) {
    throw new Error(`unknown --only value: ${only} (expected ${validTargets.join(" | ")})`);
  }

  const toolTargets = only ? CORPORA.filter((c) => c.name === only) : CORPORA;
  for (const spec of toolTargets) {
    ingest(spec, force, skipIngest);
    retrieval(spec);
  }

  // SR-Agents skill retrieval (its own ingest + subcommand shape).
  if (!only || only === "sragents") {
    ingestSragents(force, skipIngest);
    skillRetrieval();
  }

  agentCampaign(skipAgent);
  report();

  console.log("\n✓ benchmark run-all complete.");
}

try {
  main();
} catch (err) {
  console.error(`\n✗ ${(err as Error).message}`);
  process.exit(1);
}
