import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";
import "dotenv/config";
import { ratelFullAgent } from "./agents/ratel-full.js";
import { loadDataset } from "./dataset.js";
import { hashString } from "./pool.js";
import {
  buildCampaignTasks,
  type CellRow,
  type RunModel,
  runConcurrent,
  summarizeAgentRows,
} from "./runner.js";
import { runToolSelection } from "./tool-selection.js";
import type { Agent, Dataset } from "./types.js";

const AGENT_REGISTRY: Record<string, Agent> = {
  "ratel-full": ratelFullAgent,
};

interface CliConfig {
  dataset: string;
  pools: number[];
  modelId: string;
  agents: string[];
  outDir: string;
  seed: number;
  topK: number;
  /** Deterministically run only N scenarios (0 = all). For large corpora. */
  sample: number;
  /** Max agent cells in flight at once (stays under provider rate limits). */
  concurrency: number;
}

function parseArgs(argv: string[]): CliConfig {
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      args.set(key, next);
      i++;
    } else {
      args.set(key, "");
    }
  }
  const pools = (args.get("pools") ?? process.env.POOLS ?? "10,30").split(",").map(toInt);
  const agentsArg = args.get("agents") ?? process.env.AGENTS ?? "ratel-full";
  const agents =
    agentsArg.length === 0
      ? []
      : agentsArg
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
  return {
    dataset: args.get("dataset") ?? process.env.DATASET ?? "datasets/example-dataset.json",
    pools,
    modelId: args.get("model") ?? process.env.MODEL ?? "claude-sonnet-4-6",
    agents,
    outDir: args.get("out") ?? "results",
    seed: toInt(args.get("seed") ?? "0"),
    topK: toInt(args.get("top-k") ?? "5"),
    sample: toInt(args.get("sample") ?? process.env.SAMPLE ?? "0"),
    concurrency: toInt(args.get("concurrency") ?? process.env.CONCURRENCY ?? "8"),
  };
}

/** Deterministic seeded subset of scenarios (stable across runs). */
function sampleScenarios(
  scenarios: Dataset["scenarios"],
  n: number,
  seed: number,
): Dataset["scenarios"] {
  return [...scenarios]
    .map((s) => ({ s, k: hashString(`${seed}:${s.id}`) }))
    .sort((a, b) => a.k - b.k)
    .slice(0, n)
    .map((x) => x.s);
}

function toInt(s: string): number {
  const n = Number.parseInt(s, 10);
  if (Number.isNaN(n)) throw new Error(`expected integer, got "${s}"`);
  return n;
}

function resolveModel(id: string): RunModel | null {
  if (id.startsWith("claude-")) {
    if (!process.env.ANTHROPIC_API_KEY) return null;
    const anthropic = createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    return { id, model: anthropic(id) as LanguageModel };
  }
  if (id.startsWith("gpt-")) {
    if (!process.env.OPENAI_API_KEY) return null;
    const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });
    return { id, model: openai(id) as LanguageModel };
  }
  return null;
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  console.error(`config: ${JSON.stringify(cfg)}`);

  const dataset = loadDataset(cfg.dataset);
  if (cfg.sample > 0 && cfg.sample < dataset.scenarios.length) {
    const total = dataset.scenarios.length;
    dataset.scenarios = sampleScenarios(dataset.scenarios, cfg.sample, cfg.seed);
    console.error(`sampled ${dataset.scenarios.length} of ${total} scenarios (seed ${cfg.seed})`);
  }
  mkdirSync(cfg.outDir, { recursive: true });

  // 1. Tool-selection (retrieval-only) — no API key needed.
  console.error("running tool-selection benchmark…");
  const selection = runToolSelection(dataset, cfg.pools, cfg.topK, cfg.seed);
  const selectionPath = `${cfg.outDir}/tool-selection.jsonl`;
  writeFileSync(selectionPath, "");
  for (const row of selection.rows) {
    appendFileSync(selectionPath, `${JSON.stringify(row)}\n`);
  }
  console.error(`  wrote ${selection.rows.length} rows to ${selectionPath}`);

  // 2. Agent campaign — only if a model is configured.
  const runModel = resolveModel(cfg.modelId);
  const agentRowsPath = `${cfg.outDir}/agents.jsonl`;
  const agentRows: CellRow[] = [];

  if (!runModel) {
    console.error(
      `skipping agent campaign: no API key for model "${cfg.modelId}" ` +
        `(set ANTHROPIC_API_KEY for claude-*, OPENAI_API_KEY for gpt-*)`,
    );
  } else {
    writeFileSync(agentRowsPath, "");
    const append = (row: CellRow) => {
      agentRows.push(row);
      appendFileSync(agentRowsPath, `${JSON.stringify(row)}\n`);
      const tag = `[${row.arm} · ${row.scenarioId}#${row.turnIndex} · pool=${row.poolSize ?? "n/a"}]`;
      const verdict = row.pass ? "PASS" : row.result.error ? "ERROR" : "FAIL";
      const tokens = `${row.result.tokens.input}in/${row.result.tokens.output}out`;
      console.error(`${tag} ${verdict} ${tokens} ${row.result.wallMs}ms`);
    };

    const resolved: Agent[] = [];
    for (const agentId of cfg.agents) {
      const agent = AGENT_REGISTRY[agentId];
      if (!agent) {
        console.error(
          `unknown agent "${agentId}"; known: ${Object.keys(AGENT_REGISTRY).join(", ")}`,
        );
        continue;
      }
      resolved.push(agent);
    }

    const tasks = buildCampaignTasks(dataset, cfg.pools, runModel, resolved, cfg.seed);
    console.error(
      `running ${tasks.length} cells (oracle + baseline + ${resolved.length} agent(s)) ` +
        `at concurrency ${cfg.concurrency}…`,
    );
    await runConcurrent(tasks, cfg.concurrency, append);
    console.error(`  wrote ${agentRows.length} rows to ${agentRowsPath}`);
  }

  // 3. Summary.
  const summary = {
    config: cfg,
    toolSelection: selection.summary,
    agents: runModel ? summarizeAgentRows(agentRows) : null,
  };
  const summaryPath = `${cfg.outDir}/summary.json`;
  mkdirSync(dirname(summaryPath), { recursive: true });
  writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  console.error(`wrote summary to ${summaryPath}`);
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
