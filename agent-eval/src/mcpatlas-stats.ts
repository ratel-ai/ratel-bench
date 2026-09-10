// Statistics the existing modes never needed.
//
// At k=1 and n=55 the headline numbers carry real uncertainty, and the cheapest
// honest fix is to emit the interval alongside every rate rather than footnote
// it. ADR-0005 specified median + p90 + IQR for the older modes and it was never
// implemented; these are the primitives for both.

export function mean(xs: readonly number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Linear-interpolated percentile. Returns 0 for empty input, matching the
 *  convention in agent/src/report.ts. */
export function percentile(xs: readonly number[], q: number): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const i = (s.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
}

export function median(xs: readonly number[]): number {
  return percentile(xs, 0.5);
}

export function iqr(xs: readonly number[]): number {
  return percentile(xs, 0.75) - percentile(xs, 0.25);
}

export interface Interval {
  low: number;
  high: number;
}

/**
 * Wilson score interval for a proportion.
 *
 * Chosen over the normal approximation because it stays inside [0,1] and behaves
 * at the extremes — which matters here, since a 0% or 100% cell is entirely
 * plausible at n=55 and the naive interval would report zero width for it.
 */
export function wilson(successes: number, n: number, z = 1.96): Interval {
  if (n <= 0) return { low: 0, high: 0 };
  const p = successes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = p + z2 / (2 * n);
  const spread = z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n));
  return {
    low: Math.max(0, (centre - spread) / denom),
    high: Math.min(1, (centre + spread) / denom),
  };
}

export interface DeltaInterval extends Interval {
  delta: number;
  significant: boolean;
}

/**
 * Two-proportion difference interval (treatment minus baseline).
 *
 * `significant` is simply "the interval excludes 0". At n=55 per arm this will
 * often be false for a small effect, and reporting that plainly is the point —
 * it prevents a headline like "+4pp" being read as a result when it is noise.
 */
export function proportionDelta(
  treatmentSuccesses: number,
  treatmentN: number,
  baselineSuccesses: number,
  baselineN: number,
  z = 1.96,
): DeltaInterval {
  if (treatmentN <= 0 || baselineN <= 0) {
    return { delta: 0, low: 0, high: 0, significant: false };
  }
  const p1 = treatmentSuccesses / treatmentN;
  const p0 = baselineSuccesses / baselineN;
  const delta = p1 - p0;
  const se = Math.sqrt((p1 * (1 - p1)) / treatmentN + (p0 * (1 - p0)) / baselineN);
  const low = delta - z * se;
  const high = delta + z * se;
  return { delta, low, high, significant: low > 0 || high < 0 };
}

/** Percentage-point change, guarding division by zero. */
export function pctChange(from: number, to: number): number {
  return from === 0 ? 0 : ((to - from) / from) * 100;
}
