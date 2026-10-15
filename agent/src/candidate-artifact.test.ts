import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { candidateTool, loadSkillUniverse } from "./catalog-shapes.js";
import { loadScenarios } from "./corpus.js";
import { buildEmbeddingArtifact, loadEmbeddingArtifact } from "./embedding-artifact.js";
import { buildToolUniverse } from "./pool.js";
import { buildSkillCatalog, buildToolCatalog } from "./sdk/adapter.js";
import { resetForTests, selectVersion } from "./sdk/resolve.js";

type Benchmark = "bfcl" | "sragents";
interface Hit {
  id: string;
  score: number;
}
interface CandidateRow {
  scenario_id: string;
  target_pool_size: number;
  k: number;
  retrieved: Hit[];
}

const exec = promisify(execFile);
const agentDir = fileURLToPath(new URL("..", import.meta.url));
const documents = [
  { id: "files.read", name: "read_file", description: "Read a file from disk" },
  { id: "files.edit", name: "edit_file", description: "Edit the contents of a local file" },
  { id: "mail.send", name: "send_email", description: "Send an email message to a recipient" },
  { id: "travel.book", name: "book_flight", description: "Reserve an airplane flight" },
  {
    id: "weather.forecast",
    name: "weather_forecast",
    description: "Get the local weather forecast",
  },
];
let dir: string;

beforeAll(async () => {
  dir = mkdtempSync(join(agentDir, ".candidate-artifact-test-"));
  writeJsonl(
    "bfcl.jsonl",
    documents.map((document, index) => ({
      id: `bfcl-simple-${index}`,
      prompt: document.description,
      candidate_pool: [{ ...document, input_schema: { type: "object", properties: {} } }],
      gold_tools: [document.id],
    })),
  );
  writeJsonl(
    "skills.jsonl",
    documents.map((document) => ({ ...document, body: "Not indexed" })),
  );
  writeJsonl(
    "instances.jsonl",
    documents.map((document, index) => ({
      id: `skill-${index}`,
      dataset: "fixture",
      prompt: document.description,
      gold_skill_ids: [document.id],
    })),
  );
  writeFileSync(join(dir, "invalid.rat1"), "invalid artifact bytes");
  selectVersion("0.12.0");
  for (const benchmark of ["bfcl", "sragents"] as const) {
    await buildEmbeddingArtifact({
      benchmark,
      corpus: join(dir, "bfcl.jsonl"),
      skills: join(dir, "skills.jsonl"),
      output: join(dir, `${benchmark}.rat1`),
    });
  }
}, 60_000);

afterAll(() => {
  resetForTests();
  rmSync(dir, { recursive: true, force: true });
});

describe.each(["bfcl", "sragents"] as const)("%s candidate artifact CLI", (benchmark) => {
  it.each(["semantic", "hybrid"])("uses the supplied artifact for %s retrieval", async (method) => {
    await expect(
      exec(
        process.execPath,
        [
          ...candidateArgs(benchmark, method, join(dir, "unused.jsonl")),
          "--embedding-artifact",
          join(dir, "invalid.rat1"),
        ],
        { cwd: agentDir },
      ),
    ).rejects.toMatchObject({ code: 1 });
  });

  it.each(["semantic", "hybrid"] as const)(
    "has zero artifact misses for the exact %s registration shapes",
    async (method) => {
      const embeddingArtifact = {
        bytes: await loadEmbeddingArtifact(join(dir, `${benchmark}.rat1`)),
        onMiss: "error" as const,
      };
      const catalog =
        benchmark === "bfcl"
          ? await buildToolCatalog({
              method,
              embeddingArtifact,
              tools: buildToolUniverse(loadScenarios(join(dir, "bfcl.jsonl"))).map(candidateTool),
            })
          : await buildSkillCatalog({
              method,
              embeddingArtifact,
              skills: await loadSkillUniverse(join(dir, "skills.jsonl")),
            });
      const hits = await catalog.search("Read a file from disk", 1);
      expect(hits[0]).toMatchObject(
        benchmark === "bfcl" ? { toolId: "files.read" } : { skillId: "files.read" },
      );
    },
    60_000,
  );

  it.each(["semantic", "hybrid"])(
    "preserves candidate ids and order with an artifact for %s (except proven score ties)",
    async (method) => {
      const inline = join(dir, `${benchmark}-${method}-inline.jsonl`);
      const warmed = join(dir, `${benchmark}-${method}-warmed.jsonl`);
      await exec(process.execPath, candidateArgs(benchmark, method, inline), { cwd: agentDir });
      await exec(
        process.execPath,
        [
          ...candidateArgs(benchmark, method, warmed),
          "--embedding-artifact",
          join(dir, `${benchmark}.rat1`),
        ],
        { cwd: agentDir },
      );
      const inlineRows = readRows(inline);
      const warmedRows = readRows(warmed);
      expect(inlineRows[0].retrieved[0].id).toBe("files.read");
      expectRankingParity(inlineRows, warmedRows);
    },
    60_000,
  );

  it("embeds inline without reading a missing artifact on legacy SDK 0.5.0", async () => {
    const output = join(dir, `${benchmark}-legacy-candidates.jsonl`);
    const { stderr } = await exec(
      process.execPath,
      [
        ...candidateArgs(benchmark, "semantic", output, "0.5.0"),
        "--embedding-artifact",
        join(dir, "missing.rat1"),
      ],
      { cwd: agentDir },
    );
    expect(stderr).toContain("SDK 0.5.0 lacks experimentalEmbeddingArtifact; embedding inline");
    expect(readRows(output)[0].retrieved[0].id).toBe("files.read");
  }, 60_000);

  it("rejects --embedding with --embedding-artifact before building catalogs", async () => {
    await expect(
      exec(
        process.execPath,
        [
          ...candidateArgs(benchmark, "semantic", join(dir, "unused-combo.jsonl")),
          "--embedding",
          JSON.stringify({ huggingface: "Xenova/all-MiniLM-L6-v2" }),
          "--embedding-artifact",
          join(dir, `${benchmark}.rat1`),
        ],
        { cwd: agentDir },
      ),
    ).rejects.toMatchObject({
      code: 1,
      stderr: expect.stringContaining("--embedding cannot be combined with --embedding-artifact"),
    });
  });

  it("embeds missing artifact entries and keeps the full candidate pool", async () => {
    const partial = join(dir, `${benchmark}-partial.jsonl`);
    const source = join(dir, benchmark === "bfcl" ? "bfcl.jsonl" : "skills.jsonl");
    writeFileSync(
      partial,
      `${readFileSync(source, "utf8").trim().split("\n").slice(0, -1).join("\n")}\n`,
    );
    const artifact = join(dir, `${benchmark}-partial.rat1`);
    await buildEmbeddingArtifact({ benchmark, corpus: partial, skills: partial, output: artifact });
    const output = join(dir, `${benchmark}-partial-candidates.jsonl`);
    await exec(
      process.execPath,
      [...candidateArgs(benchmark, "hybrid", output), "--embedding-artifact", artifact],
      { cwd: agentDir },
    );
    const rows = readRows(output);
    const weather = rows.find(
      (row) =>
        row.scenario_id === (benchmark === "bfcl" ? "bfcl-simple-4" : "skill-4") &&
        row.target_pool_size === 5 &&
        row.k === 5,
    );
    expect(weather?.retrieved[0].id).toBe("weather.forecast");
    expect(weather?.retrieved.map((hit) => hit.id).sort()).toEqual([
      "files.edit",
      "files.read",
      "mail.send",
      "travel.book",
      "weather.forecast",
    ]);
  }, 60_000);
});

function candidateArgs(
  benchmark: Benchmark,
  method: string,
  output: string,
  sdkVersion = "0.12.0",
): string[] {
  return [
    "--import",
    "tsx",
    `src/${benchmark}-candidates.ts`,
    "--sdk-version",
    sdkVersion,
    "--retriever",
    method,
    "--pool-sizes",
    "3,5",
    "--output",
    output,
    ...(benchmark === "bfcl"
      ? ["--corpus", join(dir, "bfcl.jsonl")]
      : ["--skills", join(dir, "skills.jsonl"), "--instances", join(dir, "instances.jsonl")]),
  ];
}

function writeJsonl(name: string, rows: unknown[]): void {
  writeFileSync(join(dir, name), `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
}

function readRows(path: string): CandidateRow[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}

function expectRankingParity(inline: CandidateRow[], warmed: CandidateRow[]): void {
  const rowKey = (row: CandidateRow) => [row.scenario_id, row.target_pool_size, row.k];
  expect(warmed.map(rowKey)).toEqual(inline.map(rowKey));
  for (const [index, row] of inline.entries()) {
    const actual = warmed[index];
    const fullIndex = inline.findIndex(
      (full) =>
        full.scenario_id === row.scenario_id &&
        full.target_pool_size === row.target_pool_size &&
        full.k === full.target_pool_size,
    );
    expect(fullIndex).toBeGreaterThanOrEqual(0);
    const inlineScores = new Map(inline[fullIndex].retrieved.map((hit) => [hit.id, hit.score]));
    const warmedScores = new Map(warmed[fullIndex].retrieved.map((hit) => [hit.id, hit.score]));
    expect([...warmedScores.keys()].sort()).toEqual([...inlineScores.keys()].sort());
    expect(actual.retrieved).toHaveLength(row.retrieved.length);
    for (const [rank, expected] of row.retrieved.entries()) {
      const hit = actual.retrieved[rank];
      expect(Math.abs(hit.score - (inlineScores.get(hit.id) ?? NaN))).toBeLessThanOrEqual(1e-6);
      if (hit.id !== expected.id) {
        // An order/boundary change is allowed only within a tie in BOTH rankings.
        expect(Math.abs(expected.score - (inlineScores.get(hit.id) ?? NaN))).toBeLessThanOrEqual(
          1e-6,
        );
        expect(Math.abs(hit.score - (warmedScores.get(expected.id) ?? NaN))).toBeLessThanOrEqual(
          1e-6,
        );
      }
    }
  }
}
