import { describe, expect, it } from "vitest";
import { MAX_TIMER_MS, parsePositiveInt, parseTimerMs } from "./positive-int.js";

describe("parsePositiveInt", () => {
  it("accepts digits only (surrounding space allowed), ≥ 1", () => {
    expect(parsePositiveInt("--x", "7")).toBe(7);
    expect(parsePositiveInt("--x", " 7 ")).toBe(7);
    expect(() => parsePositiveInt("--x", " ")).toThrow('--x must be a positive integer (got " ")');
  });
});

describe("parseTimerMs", () => {
  it("also caps at Node's setTimeout limit (a longer delay would fire after 1 ms)", () => {
    expect(MAX_TIMER_MS).toBe(2 ** 31 - 1);
    expect(parseTimerMs("--timeout-ms", "2147483647")).toBe(2_147_483_647);
    expect(() => parseTimerMs("--timeout-ms", "2147483648")).toThrow(
      '--timeout-ms must be ≤ 2147483647 ms (got "2147483648")',
    );
    expect(() => parseTimerMs("--timeout-ms", "0")).toThrow(/must be a positive integer/);
  });
});
