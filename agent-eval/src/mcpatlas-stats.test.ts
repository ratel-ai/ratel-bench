import { describe, expect, it } from "vitest";
import { iqr, mean, median, percentile, proportionDelta, wilson } from "./mcpatlas-stats.js";

describe("percentiles", () => {
  it("interpolates", () => {
    expect(percentile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(median([1, 2, 3])).toBe(2);
    expect(percentile([1, 2, 3, 4, 5], 0.9)).toBeCloseTo(4.6, 10);
  });

  it("returns 0 for empty input, matching report.ts convention", () => {
    expect(percentile([], 0.5)).toBe(0);
    expect(mean([])).toBe(0);
  });

  it("iqr is p75 - p25", () => {
    expect(iqr([1, 2, 3, 4, 5])).toBe(2);
  });
});

describe("wilson", () => {
  it("brackets the observed rate", () => {
    const i = wilson(30, 55);
    expect(i.low).toBeLessThan(30 / 55);
    expect(i.high).toBeGreaterThan(30 / 55);
  });

  it("stays inside [0,1] at the extremes, unlike the normal approximation", () => {
    const zero = wilson(0, 55);
    expect(zero.low).toBe(0);
    expect(zero.high).toBeGreaterThan(0); // not a zero-width interval
    const all = wilson(55, 55);
    expect(all.high).toBeCloseTo(1, 10);
    expect(all.high).toBeLessThanOrEqual(1);
    expect(all.low).toBeLessThan(1);
  });

  it("is wide at n=55 — which is the honesty this exists for", () => {
    const i = wilson(28, 55); // ~51%
    expect(i.high - i.low).toBeGreaterThan(0.2); // >20pp
  });

  it("narrows as n grows", () => {
    const small = wilson(50, 100);
    const big = wilson(500, 1000);
    expect(big.high - big.low).toBeLessThan(small.high - small.low);
  });

  it("is 0-width for n=0 rather than NaN", () => {
    expect(wilson(0, 0)).toEqual({ low: 0, high: 0 });
  });
});

describe("proportionDelta", () => {
  it("reports the signed difference", () => {
    const d = proportionDelta(30, 55, 25, 55);
    expect(d.delta).toBeCloseTo(5 / 55, 10);
  });

  it("a small effect at n=55 is NOT significant — the guard against over-claiming", () => {
    const d = proportionDelta(30, 55, 28, 55); // +3.6pp
    expect(d.significant).toBe(false);
    expect(d.low).toBeLessThan(0);
    expect(d.high).toBeGreaterThan(0);
  });

  it("a large effect is significant", () => {
    const d = proportionDelta(50, 55, 10, 55);
    expect(d.significant).toBe(true);
    expect(d.low).toBeGreaterThan(0);
  });

  it("detects a significant negative delta too", () => {
    const d = proportionDelta(10, 55, 50, 55);
    expect(d.significant).toBe(true);
    expect(d.high).toBeLessThan(0);
  });

  it("is inert with an empty arm", () => {
    expect(proportionDelta(0, 0, 5, 10).significant).toBe(false);
  });
});
