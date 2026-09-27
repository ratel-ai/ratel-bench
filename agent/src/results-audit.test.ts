import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { summarizeBfcl } from "./bfcl-summarize.js";
import {
  type AuditRow,
  auditRows,
  mixedHarness,
  runResultsAudit,
  staleControlKeys,
} from "./results-audit.js";
import type { CellResult } from "./types.js";

const T1 = "2026-06-23T00:00:00.000Z";
const T2 = "2026-06-24T00:00:00.000Z";
const TRANSIENT = "Failed after 3 attempts. Last error: Internal server error";

function row(over: Partial<AuditRow>): AuditRow {
  return {
    scenario_id: "bfcl-multiple-0",
    arm: "control-oracle",
    model: "claude-haiku-4-5",
    run_index: 0,
    pool_size: null,
    ratel_version: "0.1.5",
    ratel_ai_core_version: "0.2.0",
    generated_at: T1,
    error: null,
    ...over,
  };
}

describe("auditRows", () => {
  it("counts by file/label/arm/model/class/ratel_version", () => {
    const counts = auditRows([
      {
        file: "agent.jsonl",
        rows: [
          row({}),
          row({ scenario_id: "bfcl-multiple-6", truncated_steps: 1 }), // no finish_reason
          row({ ratel_version: "0.1.4" }), // same group otherwise: its own count
          row({ scenario_id: "bfcl-multiple-1", error: TRANSIENT }),
          row({ scenario_id: "bfcl-multiple-2", error: TRANSIENT }),
          row({ ratel_ai_core_version: "0.3.0-rc.1", error: "run timed out after 180000ms" }),
          row({ arm: "ratel-full", finish_reason: "length", error: "x", error_class: "outcome" }),
          row({ model: "gpt-5.4-mini" }), // same group otherwise: its own count
        ],
      },
      { file: "agent-0.4.0-sparse.jsonl", rows: [row({ ratel_version: "0.4.0" })] },
    ]);
    expect(counts).toEqual([
      {
        file: "agent-0.4.0-sparse.jsonl",
        label: "0.2.0",
        arm: "control-oracle",
        model: "claude-haiku-4-5",
        error_class: "ok",
        ratel_version: "0.4.0",
        rows: 1,
        truncated: 0,
      },
      {
        file: "agent.jsonl",
        label: "0.2.0",
        arm: "control-oracle",
        model: "claude-haiku-4-5",
        error_class: "ok",
        ratel_version: "0.1.4",
        rows: 1,
        truncated: 0,
      },
      {
        file: "agent.jsonl",
        label: "0.2.0",
        arm: "control-oracle",
        model: "claude-haiku-4-5",
        error_class: "ok",
        ratel_version: "0.1.5",
        rows: 2,
        truncated: 1,
      },
      {
        file: "agent.jsonl",
        label: "0.2.0",
        arm: "control-oracle",
        model: "claude-haiku-4-5",
        error_class: "transient",
        ratel_version: "0.1.5",
        rows: 2,
        truncated: 0,
      },
      {
        file: "agent.jsonl",
        label: "0.2.0",
        arm: "control-oracle",
        model: "gpt-5.4-mini",
        error_class: "ok",
        ratel_version: "0.1.5",
        rows: 1,
        truncated: 0,
      },
      {
        file: "agent.jsonl",
        label: "0.2.0",
        arm: "ratel-full",
        model: "claude-haiku-4-5",
        error_class: "outcome",
        ratel_version: "0.1.5",
        rows: 1,
        truncated: 1,
      },
      {
        file: "agent.jsonl",
        label: "0.3.0-rc.1",
        arm: "control-oracle",
        model: "claude-haiku-4-5",
        error_class: "timeout",
        ratel_version: "0.1.5",
        rows: 1,
        truncated: 0,
      },
    ]);
  });
});

describe("staleControlKeys", () => {
  it("lists the control keys the OLD earliest-wins rule serves as errors (bfcl)", () => {
    const stale = staleControlKeys("bfcl", [
      {
        file: "agent.jsonl",
        rows: [
          row({ scenario_id: "bfcl-multiple-0", error: TRANSIENT, generated_at: T1 }),
          row({ scenario_id: "bfcl-multiple-0", generated_at: T2 }), // later good row: ignored
          row({ scenario_id: "bfcl-multiple-1", generated_at: T1 }),
          row({ scenario_id: "bfcl-multiple-1", error: TRANSIENT, generated_at: T2 }),
          row({ scenario_id: "bfcl-multiple-2", arm: "ratel-full", error: TRANSIENT }), // not cached
          row({ scenario_id: "bfcl-multiple-3", error: TRANSIENT, ratel_version: undefined }), // skipped
        ],
      },
      {
        file: "agent-0.4.0-sparse.jsonl",
        rows: [
          row({
            scenario_id: "bfcl-multiple-4",
            arm: "control-baseline",
            pool_size: 100,
            error: TRANSIENT,
          }),
          row({
            scenario_id: "bfcl-multiple-1",
            error: "gated",
            error_class: "access",
            generated_at: "2026-06-22T00:00:00.000Z",
          }),
        ],
      },
    ]);
    expect(stale.map((s) => [s.key, s.error_class, s.file])).toEqual([
      ["bfcl-multiple-0::control-oracle::claude-haiku-4-5::0", "transient", "agent.jsonl"],
      [
        "bfcl-multiple-1::control-oracle::claude-haiku-4-5::0",
        "access",
        "agent-0.4.0-sparse.jsonl",
      ],
      [
        "bfcl-multiple-4::control-baseline::claude-haiku-4-5::0::p100",
        "transient",
        "agent-0.4.0-sparse.jsonl",
      ],
    ]);
  });

  it("on a generated_at tie the first row seen wins (as the old runner's strict <)", () => {
    const errorFirst = staleControlKeys("bfcl", [
      { file: "agent.jsonl", rows: [row({ error: TRANSIENT }), row({})] },
    ]);
    expect(errorFirst.map((s) => s.key)).toEqual([
      "bfcl-multiple-0::control-oracle::claude-haiku-4-5::0",
    ]);
    const goodFirst = staleControlKeys("bfcl", [
      { file: "agent.jsonl", rows: [row({}), row({ error: TRANSIENT })] },
    ]);
    expect(goodFirst).toEqual([]);
  });

  it("uses the sragents control key (pool before run index, no ratel_version needed)", () => {
    const sr = (over: Partial<AuditRow>) =>
      row({ scenario_id: "sragents-toolqa_0", ratel_version: undefined, pool_size: 50, ...over });
    const stale = staleControlKeys("sragents", [
      {
        file: "agent.jsonl",
        rows: [sr({ arm: "control-baseline", error: "No object generated: could not parse" })],
      },
    ]);
    expect(stale.map((s) => [s.key, s.error_class])).toEqual([
      ["sragents-toolqa_0::control-baseline::claude-haiku-4-5::50::0", "outcome"],
    ]);
  });
});

describe("mixedHarness", () => {
  it("reports mixed provider/cap per label × arm × model", () => {
    const mixed = mixedHarness([
      row({ provider: "anthropic.messages" }),
      row({ provider: "amazon-bedrock", max_output_tokens: 4096 }),
      row({ arm: "ratel-full", provider: "anthropic.messages" }),
      row({ arm: "ratel-full", provider: "anthropic.messages" }),
    ]);
    expect(mixed).toEqual([
      {
        label: "0.2.0",
        arm: "control-oracle",
        model: "claude-haiku-4-5",
        providers: { "anthropic.messages": 1, "amazon-bedrock": 1 },
        caps: { unset: 1, "4096": 1 },
      },
    ]);
  });

  it.each<[string, AuditRow[], number]>([
    [
      "a cap-only mix is reported",
      [row({ provider: "a" }), row({ provider: "a", max_output_tokens: 4096 })],
      1,
    ],
    [
      "a provider-only mix is reported",
      [
        row({ provider: "a", max_output_tokens: 4096 }),
        row({ provider: "b", max_output_tokens: 4096 }),
      ],
      1,
    ],
    [
      "labels that are each homogeneous are not mixed",
      [
        row({ provider: "a" }),
        row({ ratel_ai_core_version: "0.3.0", provider: "b", max_output_tokens: 4096 }),
      ],
      0,
    ],
    [
      "models that are each homogeneous are not mixed",
      [
        row({ provider: "a" }),
        row({ model: "gpt-6-astra", provider: "b", max_output_tokens: 4096 }),
      ],
      0,
    ],
  ])("%s", (_name, rows, groups) => {
    expect(mixedHarness(rows)).toHaveLength(groups);
  });
});

describe("runResultsAudit (CLI)", () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });

  function tmpDir(): string {
    const dir = mkdtempSync(join(tmpdir(), "results-audit-"));
    dirs.push(dir);
    return dir;
  }

  function fixture(): { dir: string; agent: string; lines: string[] } {
    const dir = tmpDir();
    const agent = join(dir, "agent.jsonl");
    const lines = [
      row({ scenario_id: "bfcl-multiple-0" }),
      row({ scenario_id: "bfcl-multiple-1", error: TRANSIENT }),
      row({ scenario_id: "bfcl-multiple-2", error: "gated", error_class: "access" }),
      row({ scenario_id: "bfcl-multiple-3", error: "400", error_class: "request" }),
      row({ scenario_id: "bfcl-multiple-4", error: "run timed out after 1ms" }),
      row({ scenario_id: "bfcl-multiple-5", error: "Output blocked by content filtering policy" }),
    ].map((r) => JSON.stringify(r));
    writeFileSync(agent, `${lines.join("\n")}\n`, "utf-8");
    return { dir, agent, lines };
  }

  /** A row `summarizeBfcl` can read (it needs verdicts, tool calls and usage). */
  function bfclCell(over: Partial<AuditRow>): AuditRow {
    const cell: Partial<CellResult> = {
      category: "bfcl-multiple",
      programmatic_verdict: over.error ? "fail" : "pass",
      ast_verdict: "n/a",
      tool_calls: [],
      input_tokens: 1,
      output_tokens: 1,
      total_tokens: 2,
      dollar_cost: 0,
      wall_ms: 1,
      turns: 1,
    };
    return { ...cell, ...row(over) } as AuditRow;
  }

  function summaryOf(path: string) {
    const cells = readFileSync(path, "utf-8")
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as CellResult);
    return summarizeBfcl({ retrievalRows: [], cells, scenarios: [] }).taskSummary;
  }

  it("--drop-infra-errors --out P drops only superseded transient|access rows", () => {
    const dir = tmpDir();
    const agent = join(dir, "agent.jsonl");
    const out = join(dir, "agent.cleaned.jsonl");
    const s = (n: number) => `bfcl-multiple-${n}`;
    const lines = [
      bfclCell({ scenario_id: s(0), error: TRANSIENT }), // 0: superseded by 1 → dropped
      bfclCell({ scenario_id: s(0), generated_at: T2 }),
      bfclCell({ scenario_id: s(1), error: TRANSIENT }), // 2: final → kept
      bfclCell({ scenario_id: s(2), error: "gated", error_class: "access" }), // 3: final → kept
      bfclCell({ scenario_id: s(3), error: "400", error_class: "request" }), // 4: kept
      bfclCell({ scenario_id: s(3), error: TRANSIENT, generated_at: T2 }), // 5: supersede winner
      bfclCell({ scenario_id: s(4) }),
      bfclCell({ scenario_id: s(4), error: TRANSIENT, generated_at: T2 }), // 7: newer than its winner
      bfclCell({ scenario_id: s(5), error: TRANSIENT }), // 8: superseded by the timeout → dropped
      bfclCell({ scenario_id: s(5), error: "run timed out after 1ms", generated_at: T2 }),
      // The drop keys on label, SDK version and model too: each transient row below is final.
      bfclCell({ scenario_id: s(6), error: TRANSIENT }), // 10: other label's good row → kept
      bfclCell({ scenario_id: s(6), ratel_ai_core_version: "0.3.0-rc.1", generated_at: T2 }),
      bfclCell({ scenario_id: s(7), ratel_version: "0.1.4", error: TRANSIENT }), // 12: other SDK → kept
      bfclCell({ scenario_id: s(7), generated_at: T2 }),
      bfclCell({ scenario_id: s(8), error: TRANSIENT }), // 14: other model's good row → kept
      bfclCell({ scenario_id: s(8), model: "gpt-5.4-mini", generated_at: T2 }),
    ].map((r) => JSON.stringify(r));
    writeFileSync(agent, `${lines.join("\n")}\n\n`, "utf-8"); // trailing blank line
    const logged: string[] = [];
    runResultsAudit(
      ["--bench", "bfcl", "--agent", agent, "--drop-infra-errors", "--out", out],
      (line) => logged.push(line),
    );
    const kept = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15].map((i) => lines[i]);
    expect(readFileSync(out, "utf-8")).toBe(`${kept.join("\n")}\n`); // trailing blank line gone
    expect(logged.at(-1)).toMatch(/^dropped 2 superseded infra-errored rows; kept 6 final/);
    expect(logged.at(-1)).toContain("and 1 superseded ones newer than their cell's kept row");
    expect(readFileSync(agent, "utf-8")).toBe(`${lines.join("\n")}\n\n`); // input untouched
    // The cleanup never changes a summary: same metrics, counts and timestamps.
    expect(summaryOf(out)).toEqual(summaryOf(agent));
  });

  it("--drop-infra-errors keys sragents rows on the SR cell key", () => {
    const dir = tmpDir();
    const agent = join(dir, "agent.jsonl");
    const out = join(dir, "agent.cleaned.jsonl");
    const sr = (over: Partial<AuditRow>) =>
      row({ scenario_id: "sragents-toolqa_0", ratel_version: undefined, pool_size: 50, ...over });
    const lines = [
      sr({ error: TRANSIENT }), // superseded by the next row → dropped
      sr({ generated_at: T2 }),
      sr({ pool_size: 100, error: TRANSIENT }), // another cell: final → kept
      sr({ run_index: 1, error: TRANSIENT }), // other label's good row → kept
      sr({ run_index: 1, ratel_ai_core_version: "0.3.0-rc.1", generated_at: T2 }),
    ].map((r) => JSON.stringify(r));
    writeFileSync(agent, `${lines.join("\n")}\n`, "utf-8");
    runResultsAudit(
      ["--bench", "sragents", "--agent", agent, "--drop-infra-errors", "--out", out],
      () => {},
    );
    expect(readFileSync(out, "utf-8")).toBe(`${lines.slice(1).join("\n")}\n`);
  });

  it("--out keeps malformed lines verbatim in place, drops blank lines, logs the malformed count", () => {
    const dir = tmpDir();
    const agent = join(dir, "agent.jsonl");
    const out = join(dir, "agent.cleaned.jsonl");
    const a = JSON.stringify(row({ error: TRANSIENT })); // superseded by b → dropped
    const b = JSON.stringify(row({ generated_at: T2 }));
    const bad = '{"scenario_id":"bfcl-multiple-9","arm":';
    writeFileSync(agent, `${a}\n\n${bad}\n${b}\n\n`, "utf-8"); // blank lines mid-file and at the end
    const logged: string[] = [];
    runResultsAudit(
      ["--bench", "bfcl", "--agent", agent, "--drop-infra-errors", "--out", out],
      (line) => logged.push(line),
    );
    expect(readFileSync(out, "utf-8")).toBe(`${bad}\n${b}\n`);
    expect(logged).toContain("errors: 1 of 2 rows (transient 1); 1 malformed lines");
  });

  it("writes nothing without --out, and prints the audit", () => {
    const { dir, agent } = fixture();
    const logged: string[] = [];
    runResultsAudit(["--bench", "bfcl", "--agent", agent], (line) => logged.push(line));
    expect(readdirSync(dir)).toEqual(["agent.jsonl"]);
    const text = logged.join("\n");
    expect(text).toContain("errors: 5 of 6 rows");
    expect(text).toContain("old-rule stale control keys: 5");
    // The default cache set is the agent file itself: audited once, not twice.
    expect(logged).toContain(
      "  agent.jsonl · 0.2.0 · control-oracle · claude-haiku-4-5 · transient · 0.1.5: 1",
    );
  });

  it("--cache a,b replays the old rule across files and audits the agent file once", () => {
    const { dir, agent } = fixture();
    const cacheA = join(dir, "cache-a.jsonl");
    const early = row({ error: TRANSIENT, generated_at: "2026-06-01T00:00:00.000Z" });
    writeFileSync(cacheA, `${JSON.stringify(early)}\n`, "utf-8");
    const logged: string[] = [];
    runResultsAudit(["--bench", "bfcl", "--agent", agent, "--cache", `${cacheA},${agent}`], (l) =>
      logged.push(l),
    );
    const text = logged.join("\n");
    expect(text).toContain("old-rule stale control keys: 6"); // cache-a's earlier error wins s0
    expect(text).toContain("across 2 cache file(s)");
    expect(text).toMatch(
      /bfcl-multiple-0::control-oracle::claude-haiku-4-5::0 \[transient\] cache-a\.jsonl/,
    );
    expect(logged).toContain(
      "  agent.jsonl · 0.2.0 · control-oracle · claude-haiku-4-5 · transient · 0.1.5: 1",
    );
    expect(logged).toContain(
      "  cache-a.jsonl · 0.2.0 · control-oracle · claude-haiku-4-5 · transient · 0.1.5: 1",
    );
  });

  it("prints truncated ok groups as their truncated count, and omits clean ok groups", () => {
    const dir = tmpDir();
    const agent = join(dir, "agent.jsonl");
    const rows = [
      row({ scenario_id: "bfcl-multiple-0" }),
      row({ scenario_id: "bfcl-multiple-1", truncated_steps: 1 }),
      row({ scenario_id: "bfcl-multiple-2" }),
      row({ scenario_id: "bfcl-multiple-3", error: TRANSIENT }),
      row({ scenario_id: "bfcl-multiple-4", arm: "ratel-full" }),
    ];
    writeFileSync(agent, `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`, "utf-8");
    const logged: string[] = [];
    runResultsAudit(["--bench", "bfcl", "--agent", agent], (l) => logged.push(l));
    expect(logged).toContain(
      "  agent.jsonl · 0.2.0 · control-oracle · claude-haiku-4-5 · ok · 0.1.5: 1 truncated (of 3 ok)",
    );
    expect(logged).toContain(
      "  agent.jsonl · 0.2.0 · control-oracle · claude-haiku-4-5 · transient · 0.1.5: 1",
    );
    expect(logged.some((l) => l.includes("ratel-full"))).toBe(false);
  });

  it("rejects --drop-infra-errors without --out, and --out onto the input", () => {
    const { agent } = fixture();
    expect(() =>
      runResultsAudit(["--bench", "bfcl", "--agent", agent, "--drop-infra-errors"], () => {}),
    ).toThrow(/--out/);
    expect(() =>
      runResultsAudit(
        ["--bench", "bfcl", "--agent", agent, "--drop-infra-errors", "--out", agent],
        () => {},
      ),
    ).toThrow(/--out/);
    expect(existsSync(agent)).toBe(true);
  });

  it("rejects --out without --drop-infra-errors and writes nothing", () => {
    const { dir, agent } = fixture();
    const out = join(dir, "x.jsonl");
    expect(() =>
      runResultsAudit(["--bench", "bfcl", "--agent", agent, "--out", out], () => {}),
    ).toThrow(/--drop-infra-errors/);
    expect(existsSync(out)).toBe(false);
  });

  it("lists label cells under >1 ratel_version and the infra rows' version per label (bfcl)", () => {
    const dir = tmpDir();
    const agent = join(dir, "agent.jsonl");
    const rc = "0.3.0-rc.1";
    const rows = [
      // Label 0.3.0-rc.1 errored at SDK 0.3.0-rc.1, then re-drained at 0.1.5: two cells each.
      row({ ratel_ai_core_version: rc, ratel_version: rc, error: TRANSIENT }),
      row({ ratel_ai_core_version: rc, ratel_version: "0.1.5", generated_at: T2 }),
      row({ scenario_id: "bfcl-multiple-1", ratel_ai_core_version: rc, ratel_version: rc }),
      row({ scenario_id: "bfcl-multiple-1", ratel_ai_core_version: rc, generated_at: T2 }),
      row({ run_index: 1 }), // label 0.2.0, one version per cell: not split
      row({ scenario_id: "bfcl-multiple-2", error: TRANSIENT }),
    ];
    writeFileSync(agent, `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`, "utf-8");
    const logged: string[] = [];
    runResultsAudit(["--bench", "bfcl", "--agent", agent], (l) => logged.push(l));
    expect(logged).toContain(
      "label cells under >1 ratel_version (each version counted as its own cell): 2",
    );
    expect(logged).toContain(
      "  0.3.0-rc.1 · control-oracle · claude-haiku-4-5: 2 cells (0.3.0-rc.1 2, 0.1.5 2)",
    );
    expect(logged).toContain(
      "infra-errored rows' ratel_version per label (re-drain with --ratel-version <that>):",
    );
    expect(logged).toContain("  0.2.0: 0.1.5 1");
    expect(logged).toContain("  0.3.0-rc.1: 0.3.0-rc.1 1");
  });

  it("omits the ratel_version sections for sragents (its cell key has no SDK version)", () => {
    const dir = tmpDir();
    const agent = join(dir, "agent.jsonl");
    const rows = [row({ ratel_version: "a" }), row({ ratel_version: "b", error: TRANSIENT })];
    writeFileSync(agent, `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`, "utf-8");
    const logged: string[] = [];
    runResultsAudit(["--bench", "sragents", "--agent", agent], (l) => logged.push(l));
    expect(logged.some((l) => l.includes(">1 ratel_version"))).toBe(false);
    expect(logged.some((l) => l.includes("--ratel-version"))).toBe(false);
  });

  it("rejects an unknown --bench", () => {
    const { agent } = fixture();
    expect(() => runResultsAudit(["--bench", "mcpatlas", "--agent", agent], () => {})).toThrow(
      /--bench/,
    );
  });
});
