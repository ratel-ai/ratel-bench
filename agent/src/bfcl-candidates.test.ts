import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { sampleScenarios } from "./scenario-sample.js";

const AGENT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function scenario(index: number) {
  const name = `tool_${index}`;
  return {
    id: `bfcl-simple-simple_${index}`,
    prompt: `Call ${name} with value ${index}.`,
    candidate_pool: [
      { id: name, name, description: `Tool number ${index}.`, input_schema: { type: "object", properties: {} }, output_schema: {} },
    ],
    gold_tools: [name],
    judge_criteria: null,
    category: "bfcl-simple",
    gold_calls: [{ tool: name, args: {} }],
  };
}

describe("bfcl-candidates", () => {
  it("caps scenarios with the agent runner's seeded sample over the full tool universe", () => {
    const dir = mkdtempSync(join(tmpdir(), "bfcl-candidates-"));
    const corpus = join(dir, "bfcl.jsonl");
    const output = join(dir, "retrieval.jsonl");
    const scenarios = Array.from({ length: 6 }, (_, index) => scenario(index));
    writeFileSync(corpus, `${scenarios.map((row) => JSON.stringify(row)).join("\n")}\n`);

    execFileSync(
      process.execPath,
      ["--import", "tsx", "src/bfcl-candidates.ts", "--corpus", corpus, "--output", output, "--scenarios", "2", "--pool-sizes", "30", "--seed", "7"],
      { cwd: AGENT_DIR, stdio: "pipe" },
    );

    const rows = readFileSync(output, "utf8").trim().split("\n").map((line) => JSON.parse(line));
    const expected = sampleScenarios(scenarios as never, 2, 7).map((row) => row.id);
    expect(expected).not.toEqual(["bfcl-simple-simple_0", "bfcl-simple-simple_1"]);
    expect([...new Set(rows.map((row) => row.scenario_id))].sort()).toEqual([...expected].sort());
    expect(new Set(rows.map((row) => row.actual_pool_size))).toEqual(new Set([6]));
  });
});
