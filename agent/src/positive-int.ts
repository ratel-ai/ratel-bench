// Positive-integer parsing for CLI flags and env knobs. A leaf module (no
// imports), so the agent loop can use it without importing the CLI parser
// (which imports the runner: an import cycle).

/** Node's `setTimeout` limit: a longer delay fires after 1 ms instead. */
export const MAX_TIMER_MS = 2 ** 31 - 1;

/**
 * Parse a value that must be a positive integer (digits only, ≥ 1). Shared by
 * every integer-valued flag/knob that must not silently become `NaN`/`0`;
 * `label` (the flag or env var) names it in the error.
 */
export function parsePositiveInt(label: string, raw: string): number {
  const n = Number(raw);
  if (!/^\s*\d+\s*$/.test(raw) || !Number.isSafeInteger(n) || n < 1) {
    throw new Error(`${label} must be a positive integer (got "${raw}")`);
  }
  return n;
}

/** Like `parsePositiveInt`, but 0 is allowed (a knob's "off" or "unlimited"). */
export function parseNonNegativeInt(label: string, raw: string): number {
  const n = Number(raw);
  if (!/^\s*\d+\s*$/.test(raw) || !Number.isSafeInteger(n)) {
    throw new Error(`${label} must be a non-negative integer (got "${raw}")`);
  }
  return n;
}

/** A positive integer that becomes a timer delay, so also ≤ `MAX_TIMER_MS`. */
export function parseTimerMs(label: string, raw: string): number {
  const n = parsePositiveInt(label, raw);
  if (n > MAX_TIMER_MS) {
    throw new Error(`${label} must be ≤ ${MAX_TIMER_MS} ms (got "${raw}")`);
  }
  return n;
}
