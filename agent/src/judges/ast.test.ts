import { describe, expect, it } from "vitest";
import type { GoldCall } from "../types.js";
import { judgeAst } from "./ast.js";

const call = (toolId: string, args: Record<string, unknown>) => ({ toolId, args });

describe("judgeAst", () => {
  it("n/a when the scenario has no gold_calls", () => {
    expect(judgeAst(undefined, [call("x", {})]).verdict).toBe("n/a");
    expect(judgeAst([], [call("x", {})]).verdict).toBe("n/a");
  });

  it("passes when function and all argument values match", () => {
    const gold: GoldCall[] = [{ tool: "calc", args: { base: [10], height: [5] } }];
    expect(judgeAst(gold, [call("calc", { base: 10, height: 5 })]).verdict).toBe("pass");
  });

  it("fails (wrong tool) when the gold function was never called", () => {
    const gold: GoldCall[] = [{ tool: "calc", args: { base: [10] } }];
    const diff = judgeAst(gold, [call("other", { base: 10 })]);
    expect(diff.verdict).toBe("fail");
    expect(diff.wrong_tool).toBe(true);
  });

  it("fails on a wrong argument value (8 vs 0.08 — the percent/decimal slip)", () => {
    const gold: GoldCall[] = [{ tool: "npv", args: { discount_rate: [0.08] } }];
    const diff = judgeAst(gold, [call("npv", { discount_rate: 8 })]);
    expect(diff.verdict).toBe("fail");
    expect(diff.arg_mismatches.join(" ")).toContain("discount_rate");
  });

  it('treats a `""` in the acceptable list as optional (may be omitted)', () => {
    const gold: GoldCall[] = [{ tool: "area", args: { base: [10], unit: ["units", ""] } }];
    // `unit` omitted → still a pass because "" marks it optional.
    expect(judgeAst(gold, [call("area", { base: 10 })]).verdict).toBe("pass");
  });

  it("fails on an unexpected (extra) argument", () => {
    const gold: GoldCall[] = [{ tool: "f", args: { a: [1] } }];
    const diff = judgeAst(gold, [call("f", { a: 1, bogus: 2 })]);
    expect(diff.verdict).toBe("fail");
    expect(diff.arg_mismatches.join(" ")).toContain("bogus");
  });

  it("coerces numeric strings and is case-insensitive for strings", () => {
    const gold: GoldCall[] = [{ tool: "f", args: { n: [10], city: ["New York"] } }];
    expect(judgeAst(gold, [call("f", { n: "10", city: "new york" })]).verdict).toBe("pass");
  });

  it("matches arrays element-wise", () => {
    const gold: GoldCall[] = [{ tool: "npv", args: { cash_flows: [[-50000, 10000, 15000]] } }];
    expect(judgeAst(gold, [call("npv", { cash_flows: [-50000, 10000, 15000] })]).verdict).toBe(
      "pass",
    );
  });

  it("matches nested dicts recursively (the simple_89 shape)", () => {
    // Gold wraps each nested value in an acceptable-values list.
    const gold: GoldCall[] = [
      {
        tool: "db_fetch_records",
        args: {
          conditions: [
            { department: ["Science"], school: ["Bluebird High School", "Bluebird HS"] },
          ],
        },
      },
    ];
    // Model passed the plain dict — semantically correct; must PASS (the flat
    // comparison the throwaway script used wrongly failed this).
    const diff = judgeAst(gold, [
      call("db_fetch_records", {
        conditions: { department: "Science", school: "Bluebird High School" },
      }),
    ]);
    expect(diff.verdict).toBe("pass");
  });

  it("unwraps a gateway invoke_tool call (inner tool + inner args)", () => {
    // effectiveCalls would already unwrap; this asserts the judge accepts that shape.
    const gold: GoldCall[] = [{ tool: "calc", args: { base: [10] } }];
    expect(judgeAst(gold, [call("calc", { base: 10 })]).verdict).toBe("pass");
  });
});
