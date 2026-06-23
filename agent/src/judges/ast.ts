// AST / task-completion judge: checks the effective tool-call trace against the
// scenario's `gold_calls` — the right function AND argument values, the way
// BFCL's official `possible_answer` checker does. This is stricter than the
// selection-only `judgeProgrammatic` (which checks the function name only).
//
// BFCL ground truth gives, per argument, a LIST of acceptable values. A model
// argument matches if it equals one acceptable value (with int/float and
// numeric-string coercion, and case-insensitive string compare). A list that
// contains "" marks the argument OPTIONAL (the model may omit it). Nested dicts
// are matched recursively — the dict's values are themselves acceptable-value
// lists (this is the case the naive flat comparison gets wrong). Arrays are
// matched element-wise.
//
// This ports BFCL's core rules; it is not byte-identical to the leaderboard
// harness (deeply nested generics and some exotic Python coercions are out of
// scope). Single-call matching only — `simple`/`multiple` have one gold call.

import type { GoldCall, ProgrammaticVerdict } from "../types.js";

export interface AstDiff {
  /** `n/a` when the scenario carries no `gold_calls` (non-BFCL corpora). */
  verdict: ProgrammaticVerdict;
  /** A gold call's function was never invoked. */
  wrong_tool: boolean;
  /** Human-readable per-argument failure reasons (for debugging). */
  arg_mismatches: string[];
}

interface ObservedCall {
  toolId: string;
  args: Record<string, unknown>;
}

function isPlainObject(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

/** Numeric view of a value, or null if it isn't number-like (bools excluded). */
function asNumber(x: unknown): number | null {
  if (typeof x === "number") return Number.isFinite(x) ? x : null;
  if (typeof x === "string") {
    const t = x.trim();
    if (t !== "" && Number.isFinite(Number(t))) return Number(t);
  }
  return null;
}

/** Does a model value equal one specific acceptable value? */
function matchOne(value: unknown, acceptable: unknown): boolean {
  if (value === acceptable) return true;
  if (isPlainObject(acceptable) && isPlainObject(value)) {
    // Nested dict: the acceptable dict maps key → acceptable-value list.
    return dictMatches(acceptable, value);
  }
  if (Array.isArray(acceptable) && Array.isArray(value)) {
    return acceptable.length === value.length && value.every((v, i) => matchOne(v, acceptable[i]));
  }
  const nv = asNumber(value);
  const na = asNumber(acceptable);
  if (nv !== null && na !== null) return nv === na;
  if (typeof value === "string" && typeof acceptable === "string") {
    return value.trim().toLowerCase() === acceptable.trim().toLowerCase();
  }
  return false;
}

/** A model value matches if it equals any value in the acceptable list. */
function valueMatches(value: unknown, acceptableList: unknown): boolean {
  const list = Array.isArray(acceptableList) ? acceptableList : [acceptableList];
  return list.some((a) => matchOne(value, a));
}

/** A list that includes "" means the argument is optional (may be omitted). */
function isOptional(acceptableList: unknown): boolean {
  return Array.isArray(acceptableList) && acceptableList.some((a) => a === "");
}

/**
 * Match one model arg object against a gold arg spec (`{param: [acceptable...]}`).
 * Returns the failure reasons; empty ⇒ match. Used both at the top level and
 * recursively for nested dict arguments.
 */
function dictDiff(goldArgs: Record<string, unknown>, obsArgs: Record<string, unknown>): string[] {
  const reasons: string[] = [];
  const goldKeys = new Set(Object.keys(goldArgs));
  for (const k of Object.keys(obsArgs)) {
    if (!goldKeys.has(k)) reasons.push(`unexpected arg '${k}'`);
  }
  for (const [k, acceptable] of Object.entries(goldArgs)) {
    if (k in obsArgs) {
      if (!valueMatches(obsArgs[k], acceptable)) {
        reasons.push(`'${k}'=${JSON.stringify(obsArgs[k])} not in ${JSON.stringify(acceptable)}`);
      }
    } else if (!isOptional(acceptable)) {
      reasons.push(`missing required '${k}'`);
    }
  }
  return reasons;
}

function dictMatches(goldArgs: Record<string, unknown>, obsArgs: Record<string, unknown>): boolean {
  return dictDiff(goldArgs, obsArgs).length === 0;
}

/**
 * Task-completion verdict: every gold call must be satisfied by some observed
 * call (right function + matching arguments). `n/a` when there is no argument
 * ground truth (non-BFCL corpora). Extra observed calls to other tools are
 * ignored (lenient, per-call matching like BFCL).
 */
export function judgeAst(goldCalls: GoldCall[] | undefined, observed: ObservedCall[]): AstDiff {
  if (!goldCalls || goldCalls.length === 0) {
    return { verdict: "n/a", wrong_tool: false, arg_mismatches: [] };
  }
  const mismatches: string[] = [];
  let wrongTool = false;
  let allMatched = true;

  for (const gold of goldCalls) {
    const candidates = observed.filter((o) => o.toolId === gold.tool);
    if (candidates.length === 0) {
      wrongTool = true;
      allMatched = false;
      mismatches.push(`tool '${gold.tool}' not called`);
      continue;
    }
    // Pass if any candidate call to this tool matches all args. If none match,
    // surface the closest candidate's reasons (the one with the fewest).
    let best: string[] | null = null;
    let matched = false;
    for (const cand of candidates) {
      const reasons = dictDiff(gold.args, cand.args);
      if (reasons.length === 0) {
        matched = true;
        break;
      }
      if (best === null || reasons.length < best.length) best = reasons;
    }
    if (!matched) {
      allMatched = false;
      if (best) mismatches.push(...best.map((r) => `${gold.tool}: ${r}`));
    }
  }

  return {
    verdict: allMatched ? "pass" : "fail",
    wrong_tool: wrongTool,
    arg_mismatches: mismatches,
  };
}

/**
 * Argument recall: across each gold call's **required** arguments (acceptable
 * list without `""`), the fraction the model supplied with an acceptable value —
 * a partial-credit complement to the all-or-nothing AST verdict. `0` when the
 * gold tool wasn't called; `1` for a gold call with no required args (as long as
 * the tool was called); `null` when there's no argument ground truth. For
 * several candidate calls to the gold tool, the best-matching one wins; multiple
 * gold calls are averaged.
 */
export function astArgRecall(
  goldCalls: GoldCall[] | undefined,
  observed: ObservedCall[],
): number | null {
  if (!goldCalls || goldCalls.length === 0) return null;
  const perCall: number[] = [];
  for (const gold of goldCalls) {
    const candidates = observed.filter((o) => o.toolId === gold.tool);
    if (candidates.length === 0) {
      perCall.push(0);
      continue;
    }
    const required = Object.entries(gold.args).filter(([, acc]) => !isOptional(acc));
    if (required.length === 0) {
      perCall.push(1);
      continue;
    }
    let best = 0;
    for (const cand of candidates) {
      const matched = required.filter(
        ([k, acc]) => k in cand.args && valueMatches(cand.args[k], acc),
      ).length;
      best = Math.max(best, matched / required.length);
    }
    perCall.push(best);
  }
  return perCall.reduce((a, b) => a + b, 0) / perCall.length;
}
