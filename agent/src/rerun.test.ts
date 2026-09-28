import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  type AttemptRow,
  capGuardRows,
  countResume,
  type DoneSummary,
  doneLines,
  emptyResumeCounts,
  formatDoneLine,
  newRunTally,
  nextAttempt,
  planResume,
  RERUN_FLAGS,
  rerunOutcome,
  rerunSettingsLine,
  resumeLine,
  runExitCode,
  runRounds,
  summarizeRunCoverage,
  tallyRow,
} from "./rerun.js";

const summary = (over: Partial<DoneSummary> = {}): DoneSummary => ({
  cells_run: 30,
  cells_cached: 4,
  cells_skipped: 2,
  total_dollars: 1.23456,
  stopped_reason: "completed",
  cap_hit: false,
  retries: 7,
  throttled_retries: 5,
  errors: 3,
  requeued: 2,
  exhausted: 1,
  aborted: {},
  ...over,
});

describe("formatDoneLine", () => {
  it("labels known spend as a lower bound when attempt accounting is incomplete", () => {
    const line = formatDoneLine(
      summary({
        spend: {
          attempts: 3,
          unresolved: 1,
          unknown: 1,
          untrackedRows: 0,
          knownUsd: 0.5,
          completeness: "partial",
        },
      }),
    );
    expect(line).toContain("spent (known lower bound; 1 unresolved, 1 unknown");
  });
  it("labels a known subtotal when some cells have unknown cost", () => {
    expect(formatDoneLine(summary({ unknown_cost_cells: 2 }))).toContain(
      "$1.2346 spent, 2 unknown-cost cells, stopped=completed",
    );
  });
  it("matches /^done: \\d+ cells run/ and /, \\$([0-9.]+) spent/ and appends counts", () => {
    const line = formatDoneLine(summary());
    // The bench-run.sh contract: its grep and its sed on the spend.
    expect(line).toMatch(/^done: [0-9]+ cells run/);
    expect(line.match(/, \$([0-9.]+) spent/)?.[1]).toBe("1.2346");
    expect(line).toBe(
      "done: 30 cells run, 4 cached, 2 skipped, $1.2346 spent, stopped=completed, " +
        "7 retries (5 throttled), 3 errors, 2 re-queued, 1 exhausted",
    );
  });

  it("an abort outranks the cap in stopped=, which still names the cap (fatal+global_cap)", () => {
    expect(formatDoneLine(summary({ stopped_reason: "global_cap", cap_hit: true }))).toMatch(
      /, stopped=global_cap, /,
    );
    const line = formatDoneLine(summary({ stopped_reason: "fatal", cap_hit: true }));
    expect(line).toMatch(/, stopped=fatal\+global_cap, 7 retries/);
    expect(line).toMatch(/^done: 30 cells run/);
    expect(line.match(/, (\d+) errors?\b/)?.[1]).toBe("3");
  });

  it("doneLines adds one `aborted: <model> — <reason>` line per aborted model", () => {
    const lines = doneLines(
      summary({
        stopped_reason: "fatal",
        aborted: {
          "gpt-6-astra": { reason: "fatal", detail: "model not available for this account" },
          haiku: { reason: "error_circuit", detail: "10 consecutive transient/access errors" },
        },
      }),
    );
    expect(lines).toEqual([
      expect.stringMatching(/^done: 30 cells run, .* stopped=fatal, /),
      "aborted: gpt-6-astra — fatal: model not available for this account",
      "aborted: haiku — error_circuit: 10 consecutive transient/access errors",
    ]);
  });

  it("prints structured terminal coverage and exits nonzero for incomplete work", () => {
    const incomplete = summary({
      stopped_reason: "global_cap",
      cap_hit: true,
      coverage: {
        requested: 4,
        completed: 2,
        failed: 0,
        reused: 1,
        skipped: 1,
        status: "budget_limited",
      },
    });
    expect(doneLines(incomplete)[1]).toBe(
      "coverage: status=budget_limited, requested=4, completed=2, failed=0, reused=1, skipped=1",
    );
    expect(runExitCode(incomplete)).toBe(2);
  });
});

describe("summarizeRunCoverage", () => {
  const key = (row: { key: string }) => row.key;

  it("classifies final unique cells after retries and reconciles the request", () => {
    const coverage = summarizeRunCoverage(
      ["success", "failed", "reused", "missing"],
      [
        { key: "success", error: "Overloaded" },
        { key: "success", error: null },
        { key: "failed", error: "bad output" },
        { key: "reused", error: null, cache_source: "reused" as const },
      ],
      key,
      "global_cap",
    );
    expect(coverage).toEqual({
      requested: 4,
      completed: 1,
      failed: 1,
      reused: 1,
      skipped: 1,
      status: "budget_limited",
    });
  });

  it("never calls all-error or interrupted work completed", () => {
    expect(summarizeRunCoverage(["a"], [{ key: "a", error: "bad" }], key, "completed").status).toBe(
      "failed",
    );
    expect(summarizeRunCoverage(["a"], [], key, "interrupted").status).toBe("cancelled");
  });
});

describe("resumeLine", () => {
  it("lists the non-zero classes; null when nothing was re-queued or exhausted", () => {
    expect(resumeLine({ requeued: { access: 9 }, exhausted: 0 })).toBe(
      "resume: 9 re-queued (access 9); 0 exhausted",
    );
    expect(resumeLine({ requeued: { request: 1, transient: 2 }, exhausted: 3 })).toBe(
      "resume: 3 re-queued (transient 2, request 1); 3 exhausted",
    );
    expect(resumeLine({ requeued: {}, exhausted: 2 })).toBe("resume: 0 re-queued; 2 exhausted");
    expect(resumeLine({ requeued: {}, exhausted: 0 })).toBeNull();
  });
});

describe("planResume", () => {
  const row = (key: string, over: Partial<AttemptRow> = {}): AttemptRow & { key: string } => ({
    key,
    error: null,
    cache_source: "live",
    ...over,
  });
  const keyOf = (r: { key: string }) => r.key;
  const overloaded = { error: "Overloaded" };

  it("a final row anywhere completes the key, before or after its errors", () => {
    const plan = planResume(
      [row("a", overloaded), row("a"), row("b"), row("b", overloaded)],
      keyOf,
      { policy: "infra", maxAttempts: 3 },
    );
    expect([...plan.completed].sort()).toEqual(["a", "b"]);
    expect(plan.requeue.size).toBe(0);
  });

  it("re-queues with the last row's class; attempts count live rows only", () => {
    const plan = planResume(
      [
        row("a", { error: "Overloaded", cache_source: "reused" }),
        row("a", { error: "tools: too many", error_class: "request" }),
      ],
      keyOf,
      { policy: "infra", maxAttempts: 2 },
    );
    expect(plan.requeue.get("a")).toBe("request");
    expect(plan.attempts.get("a")).toBe(1);
    expect(plan.completed.has("a")).toBe(false);
  });
});

describe("capGuardRows", () => {
  const keyOf = (r: { key: string }) => r.key;
  const request = { error: "tools: too many tools", error_class: "request" as const };
  const timeout = { error: "run timed out after 1000ms" };
  it("keeps each key's surviving row, minus rerunnable ones this run re-queues", () => {
    const rows = [
      { key: "requeued", ...request },
      { key: "outside", ...request }, // re-queueable, but not by this run
      { key: "fixed", ...request },
      { key: "fixed", error: null }, // a later success supersedes the request row
      { key: "timeout", ...timeout }, // re-queued under `all`, still a final row
    ];
    const kept = capGuardRows(rows, keyOf, new Set(["requeued", "fixed", "timeout"]));
    expect(kept.map((r) => [r.key, r.error])).toEqual([
      ["outside", request.error],
      ["fixed", null],
      ["timeout", timeout.error],
    ]);
  });
});

describe("countResume / nextAttempt", () => {
  it("counts re-queued (by class) and exhausted keys; the next attempt follows the live ones", () => {
    const plan = planResume(
      [
        { key: "a", error: "Overloaded" },
        { key: "b", error: "Overloaded" },
        { key: "b", error: "Overloaded" },
        { key: "c", error: null },
      ],
      (r) => r.key,
      { policy: "infra", maxAttempts: 2 },
    );
    const counts = emptyResumeCounts();
    for (const key of ["a", "b", "c", "d"]) countResume(plan, key, counts);
    expect(counts).toEqual({ requeued: { transient: 1 }, exhausted: 1 });
    expect(["a", "b", "c", "d"].map((k) => nextAttempt(plan, k))).toEqual([2, 3, 2, 1]);
  });
});

describe("rerunOutcome", () => {
  const overloaded = { error: "Overloaded" };
  it.each([
    [overloaded, 5, 0, "retry"], // 0 = unlimited
    [overloaded, 2, 3, "retry"],
    [overloaded, 3, 3, "exhausted"],
    [{ error: null }, 1, 3, "final"],
    [{ error: "run timed out after 1000ms" }, 1, 3, "final"],
  ] as const)("%o at attempt %i, max %i → %s", (row, attempt, maxAttempts, outcome) => {
    expect(rerunOutcome(row, attempt, { policy: "infra", maxAttempts })).toBe(outcome);
  });
});

describe("tallyRow", () => {
  it("sums retries and throttled retries; counts errored rows", () => {
    const tally = newRunTally();
    tallyRow(tally, { error: "x", retries: 3, throttled_retries: 2 });
    tallyRow(tally, { error: null, retries: 1 });
    expect(tally).toMatchObject({ retries: 4, throttled_retries: 2, errors: 1 });
  });
});

describe("runExitCode", () => {
  it("is 2 when the breaker aborted a model, else 0 (a cap stop is not a failure)", () => {
    expect(runExitCode(summary())).toBe(0);
    expect(runExitCode(summary({ stopped_reason: "global_cap", cap_hit: true }))).toBe(0);
    for (const reason of ["fatal", "error_circuit"] as const) {
      const aborted = { m: { reason, detail: "x" } };
      expect(runExitCode(summary({ stopped_reason: reason, aborted }))).toBe(2);
    }
  });
});

describe("runRounds", () => {
  it("passes the next attempt and stops when nothing is left", async () => {
    const seen: number[][] = [];
    const requeued = await runRounds(
      [{ attempt: 1 }, { attempt: 1 }],
      { rounds: 3, delayMs: 0, sleep: async () => {} },
      async (tasks) => {
        seen.push(tasks.map((t) => t.attempt));
        return tasks.slice(1);
      },
      () => true,
    );
    expect(seen).toEqual([[2, 2], [3]]);
    expect(requeued).toBe(3);
  });
});

describe("rerunSettingsLine", () => {
  it("echoes the rerun knobs at startup", () => {
    expect(rerunSettingsLine({ policy: "infra", maxAttempts: 3, rounds: 1, delayMs: 60_000 })).toBe(
      "rerun: --retry-errors infra, --max-attempts 3, --retry-rounds 1 (60s apart)",
    );
    expect(rerunSettingsLine({ policy: "all", maxAttempts: 0, rounds: 0, delayMs: 0 })).toMatch(
      /--max-attempts unlimited/,
    );
  });
});

// ratel-bench-aws `scripts/bench-flags.mjs` passes a flag only when the file that
// parses it spells it as a quoted literal (`"--flag"`), else refuses the knob.
describe("ratel-bench-aws bench-flags detection contract", () => {
  const source = (file: string) => readFileSync(new URL(`./${file}`, import.meta.url), "utf-8");
  it.each([
    "cli-args.ts",
    "sragents-select.ts",
  ])("%s spells every rerun flag and --timeout-ms as a quoted literal", (file) => {
    const text = source(file);
    for (const flag of [...RERUN_FLAGS, "--timeout-ms"]) expect(text).toContain(`"${flag}"`);
  });
});
