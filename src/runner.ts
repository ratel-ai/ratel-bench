import type { LanguageModel } from "ai";
import { baselineAgent } from "./agents/baseline.js";
import { initOracle } from "./agents/oracle.js";
import { buildScenarioPool } from "./pool.js";
import type { Agent, AgentRunResult, Dataset } from "./types.js";

export interface CellRow {
  scenarioId: string;
  turnIndex: number;
  arm: string;
  modelId: string;
  /** `null` for pool-size-agnostic arms (oracle). */
  poolSize: number | null;
  catalogSize: number;
  expectedTool: string;
  pass: boolean;
  result: AgentRunResult;
}

export type AppendRow = (row: CellRow) => void;

export interface RunModel {
  id: string;
  model: LanguageModel;
}

/** A unit of work: run one scenario (under one arm/pool) and return its rows. */
export type CellTask = () => Promise<CellRow[]>;

/** Oracle: pool-agnostic, only the gold tools exposed. One task per scenario. */
function oracleTasks(dataset: Dataset, model: RunModel): CellTask[] {
  return dataset.scenarios.map((scenario) => async () => {
    const agent = initOracle({ scenario, allTools: dataset.tools, model: model.model });
    const catalogSize = new Set(scenario.turns.map((t) => t.expectedTool)).size;
    const rows: CellRow[] = [];
    for (let i = 0; i < scenario.turns.length; i++) {
      const turn = scenario.turns[i];
      const result = await agent.run(turn.input);
      rows.push(
        makeRow(scenario.id, i, "oracle", model.id, null, catalogSize, turn.expectedTool, result),
      );
    }
    return rows;
  });
}

/** One task per (pool, scenario): build the scenario's pool, run its turns. */
function agentTasks(
  agent: Agent,
  dataset: Dataset,
  pools: number[],
  model: RunModel,
  seed: number,
): CellTask[] {
  const tasks: CellTask[] = [];
  for (const poolSize of pools) {
    for (const scenario of dataset.scenarios) {
      tasks.push(async () => {
        const pool = buildScenarioPool(dataset.tools, scenario, poolSize, seed);
        const instance = agent.init({ toolPool: pool, model: model.model });
        const rows: CellRow[] = [];
        for (let i = 0; i < scenario.turns.length; i++) {
          const turn = scenario.turns[i];
          const result = await instance.run(turn.input);
          rows.push(
            makeRow(
              scenario.id,
              i,
              agent.id,
              model.id,
              poolSize,
              pool.length,
              turn.expectedTool,
              result,
            ),
          );
        }
        return rows;
      });
    }
  }
  return tasks;
}

/**
 * Every cell of a campaign as an independent task: oracle, the baseline arm, and
 * each configured non-control agent. Arms all extend the same base loop — they
 * differ only in the tool surface they assemble — so they share one task list
 * and one concurrency cap.
 */
export function buildCampaignTasks(
  dataset: Dataset,
  pools: number[],
  model: RunModel,
  agents: Agent[],
  seed = 0,
): CellTask[] {
  const tasks: CellTask[] = [...oracleTasks(dataset, model)];
  for (const agent of [baselineAgent, ...agents]) {
    tasks.push(...agentTasks(agent, dataset, pools, model, seed));
  }
  return tasks;
}

/**
 * Run tasks with a fixed concurrency cap. Cells are independent, so this just
 * keeps `concurrency` of them in flight at once (bounded to stay under provider
 * rate limits). Rows are appended as each task resolves, so the stream is
 * interleaved across arms/pools — group in analysis. A task that throws is
 * logged and skipped rather than aborting the whole campaign.
 */
export async function runConcurrent(
  tasks: CellTask[],
  concurrency: number,
  append: AppendRow,
): Promise<void> {
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < tasks.length) {
      const idx = next++;
      try {
        for (const row of await tasks[idx]()) append(row);
      } catch (err) {
        console.error(`task ${idx} threw: ${(err as Error).message ?? String(err)}`);
      }
    }
  };
  const lanes = Math.max(1, Math.min(concurrency, tasks.length));
  await Promise.all(Array.from({ length: lanes }, () => worker()));
}

function makeRow(
  scenarioId: string,
  turnIndex: number,
  arm: string,
  modelId: string,
  poolSize: number | null,
  catalogSize: number,
  expectedTool: string,
  result: AgentRunResult,
): CellRow {
  return {
    scenarioId,
    turnIndex,
    arm,
    modelId,
    poolSize,
    catalogSize,
    expectedTool,
    pass: result.effectiveToolIds.includes(expectedTool),
    result,
  };
}

export interface AgentSummary {
  byArmAndPool: Record<
    string,
    {
      arm: string;
      poolSize: number | null;
      cells: number;
      passes: number;
      passRate: number;
      meanInputTokens: number;
      meanOutputTokens: number;
      meanTotalTokens: number;
      meanWallMs: number;
      errors: number;
    }
  >;
}

export function summarizeAgentRows(rows: CellRow[]): AgentSummary {
  const out: AgentSummary = { byArmAndPool: {} };
  const groups = new Map<string, CellRow[]>();
  for (const row of rows) {
    const key = `${row.arm}::${row.poolSize ?? "null"}`;
    const list = groups.get(key) ?? [];
    list.push(row);
    groups.set(key, list);
  }
  for (const [key, list] of groups) {
    const passes = list.filter((r) => r.pass).length;
    const errors = list.filter((r) => r.result.error !== null).length;
    out.byArmAndPool[key] = {
      arm: list[0].arm,
      poolSize: list[0].poolSize,
      cells: list.length,
      passes,
      passRate: list.length === 0 ? 0 : passes / list.length,
      meanInputTokens: mean(list.map((r) => r.result.tokens.input)),
      meanOutputTokens: mean(list.map((r) => r.result.tokens.output)),
      meanTotalTokens: mean(list.map((r) => r.result.tokens.total)),
      meanWallMs: mean(list.map((r) => r.result.wallMs)),
      errors,
    };
  }
  return out;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
