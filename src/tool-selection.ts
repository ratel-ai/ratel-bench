import { buildCatalog } from "./catalog.js";
import { buildScenarioPool } from "./pool.js";
import type { Dataset } from "./types.js";

export interface ToolSelectionRow {
  scenarioId: string;
  turnIndex: number;
  poolSize: number;
  mode: "input-only" | "expected-query";
  query: string;
  expectedTool: string;
  topK: Array<{ toolId: string; position: number }>;
  score: number;
}

export interface ToolSelectionSummary {
  byPool: Record<
    string,
    {
      inputOnlyMeanScore: number;
      queryMeanScore: number;
      turns: number;
    }
  >;
}

export function runToolSelection(
  dataset: Dataset,
  pools: number[],
  topK = 5,
  seed = 0,
): { rows: ToolSelectionRow[]; summary: ToolSelectionSummary } {
  const rows: ToolSelectionRow[] = [];

  for (const poolSize of pools) {
    for (const scenario of dataset.scenarios) {
      const pool = buildScenarioPool(dataset.tools, scenario, poolSize, seed);
      const catalog = buildCatalog(pool);

      for (let i = 0; i < scenario.turns.length; i++) {
        const turn = scenario.turns[i];
        const lastUser = lastUserMessage(turn.input.messages);

        rows.push(
          scoreSearch(catalog, lastUser, turn.expectedTool, topK, {
            scenarioId: scenario.id,
            turnIndex: i,
            poolSize,
            mode: "input-only",
          }),
        );

        rows.push(
          scoreSearch(catalog, turn.expectedQuery, turn.expectedTool, topK, {
            scenarioId: scenario.id,
            turnIndex: i,
            poolSize,
            mode: "expected-query",
          }),
        );
      }
    }
  }

  const summary: ToolSelectionSummary = { byPool: {} };
  for (const poolSize of pools) {
    const pool = rows.filter((r) => r.poolSize === poolSize);
    const inputOnly = pool.filter((r) => r.mode === "input-only");
    const query = pool.filter((r) => r.mode === "expected-query");
    summary.byPool[String(poolSize)] = {
      inputOnlyMeanScore: mean(inputOnly.map((r) => r.score)),
      queryMeanScore: mean(query.map((r) => r.score)),
      turns: inputOnly.length,
    };
  }

  return { rows, summary };
}

function scoreSearch(
  catalog: ReturnType<typeof buildCatalog>,
  query: string,
  expectedTool: string,
  topK: number,
  meta: Omit<ToolSelectionRow, "query" | "expectedTool" | "topK" | "score">,
): ToolSelectionRow {
  const hits = catalog.search(query, topK);
  const topKList = hits.map((h, idx) => ({ toolId: h.toolId, position: idx + 1 }));
  const found = topKList.find((h) => h.toolId === expectedTool);
  const score = found ? 1 / found.position : 0;
  return { ...meta, query, expectedTool, topK: topKList, score };
}

function lastUserMessage(messages: { role: string; content: string }[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") return messages[i].content;
  }
  return "";
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
