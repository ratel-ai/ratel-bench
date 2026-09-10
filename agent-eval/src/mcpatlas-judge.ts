// Task-success scoring: deterministic screen first, LLM judge for the residual.
//
// MCP-Atlas scores GTFA_CLAIMS at 0 / 0.5 / 1 and passes a task at >=75%
// coverage. None of the judges in agent/src/judges fit: judgeProgrammatic
// compares tool ids, judgeAst needs possible_answer value lists (MCP-Atlas has
// concrete recorded values, not acceptable-value sets — which is why ast_verdict
// is permanently n/a here), and judgeLLM returns one holistic verdict, discarding
// the partial credit that is the whole point of the rubric.
//
// The screen (mcpatlas-claim-match) decides the claims where a wrong call is
// implausible; only the rest cost a model call. Two invariants hold regardless of
// which path scored a claim:
//   - the judge never sees the trajectory, the gold tools, or the ARM label
//   - the verdict is computed in code from the scores, never asserted by the model

import { generateObject, type LanguageModel } from "ai";
import { z } from "zod";
import { type ClaimScreen, screenClaims } from "./mcpatlas-claim-match.js";
import type { ClaimRubricResult, ClaimScore, JudgeVerdict } from "./mcpatlas-types.js";

export const DEFAULT_PASS_THRESHOLD = 0.75;
export const DEFAULT_PARTIAL_THRESHOLD = 0.4;

const ClaimScoresSchema = z.object({
  scores: z.array(
    z.object({
      claim_index: z.number().int(),
      score: z.union([z.literal(0), z.literal(0.5), z.literal(1)]),
      evidence: z.string(),
      explanation: z.string(),
    }),
  ),
});

export const SYSTEM = [
  "You score an AI assistant's answer against a list of atomic factual claims.",
  "You will be given:",
  "  1. TASK — what the user asked.",
  "  2. CLAIMS — numbered statements a correct answer must support.",
  "  3. ANSWER — the assistant's final text.",
  "For EACH claim, return a score:",
  "  1   = the answer clearly supports the claim",
  "  0.5 = the answer partially supports it (right subject, imprecise or incomplete)",
  "  0   = the answer does not support it, contradicts it, or omits it",
  "Quote the span of ANSWER that justifies your score in `evidence`; use an empty",
  "string when nothing supports the claim.",
  "Wording, formatting and unit differences are fine — judge the substance.",
  "An answer that states the OPPOSITE of a claim scores 0, never 1.",
  "You do NOT see which tools were called or how the answer was produced.",
  "Score every claim you are given, once, using the supplied claim_index.",
].join("\n");

export function buildJudgePrompt(
  task: string,
  claims: ReadonlyArray<{ index: number; text: string }>,
  answer: string,
): string {
  return [
    "TASK:",
    task,
    "",
    "CLAIMS:",
    ...claims.map((c) => `[${c.index}] ${c.text}`),
    "",
    "ANSWER:",
    answer || "(the assistant produced no final text)",
  ].join("\n");
}

/** coverage -> verdict. Computed here, never delegated to the model. */
export function verdictFor(
  coverage: number | null,
  passThreshold = DEFAULT_PASS_THRESHOLD,
  partialThreshold = DEFAULT_PARTIAL_THRESHOLD,
): JudgeVerdict {
  if (coverage === null) return "n/a";
  if (coverage >= passThreshold) return "pass";
  if (coverage >= partialThreshold) return "partial";
  return "fail";
}

/** Screen verdicts as scores. Only called for non-ambiguous screens. */
export function screenScore(s: ClaimScreen): 0 | 1 {
  return s.verdict === "supported" ? 1 : 0;
}

export interface JudgeClaimsArgs {
  taskId: string;
  /** MCP-Atlas PROMPT. */
  prompt: string;
  /** GTFA_CLAIMS, verbatim and in order. */
  claims: readonly string[];
  /** The assistant's final text. Nothing else about the run is passed in. */
  finalText: string;
  /** Omit to run screen-only (no model calls, fully deterministic). */
  model?: LanguageModel;
  passThreshold?: number;
  partialThreshold?: number;
  /** Force every claim through the LLM, ignoring the screen. Used by the
   *  calibration path to measure screen-vs-judge agreement. */
  judgeAll?: boolean;
  /** Injection point for tests. */
  generate?: typeof generateObject;
}

export interface McpAtlasRubricResult extends ClaimRubricResult {
  /** Per-claim provenance: which path produced each score. */
  scored_by: Array<"screen" | "llm" | "unscored">;
  screens: ClaimScreen[];
  claims_auto_scored: number;
  claims_sent_to_llm: number;
  /** Share the screen decided without a model. */
  auto_rate: number;
}

/**
 * Score one task.
 *
 * Cost shape: one model call per task at most, containing only the ambiguous
 * claims. A task whose claims are all screen-decidable costs nothing. When no
 * `model` is supplied, ambiguous claims are left `unscored` and coverage is
 * `null` — deliberately not 0, because "we could not score this" and "the answer
 * was wrong" are different findings.
 */
export async function judgeClaims(args: JudgeClaimsArgs): Promise<McpAtlasRubricResult> {
  const started = Date.now();
  const {
    taskId,
    prompt,
    claims,
    finalText,
    model,
    passThreshold = DEFAULT_PASS_THRESHOLD,
    partialThreshold = DEFAULT_PARTIAL_THRESHOLD,
    judgeAll = false,
    generate = generateObject,
  } = args;

  const { screens } = screenClaims(claims, taskId, finalText);
  const needLlm = judgeAll
    ? screens.map((_, i) => i)
    : screens.map((s, i) => (s.verdict === "ambiguous" ? i : -1)).filter((i) => i >= 0);

  const scores: Array<number | null> = screens.map((s, i) =>
    needLlm.includes(i) ? null : screenScore(s),
  );
  const scoredBy: Array<"screen" | "llm" | "unscored"> = screens.map((_, i) =>
    needLlm.includes(i) ? "unscored" : "screen",
  );
  const evidence: string[] = screens.map((s) =>
    s.verdict === "supported" ? s.matched.join(", ") : "",
  );
  const explanation: string[] = screens.map((s) => s.reason);

  let judgeError: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;

  if (needLlm.length && model) {
    try {
      const res = await generate({
        model,
        schema: ClaimScoresSchema,
        system: SYSTEM,
        prompt: buildJudgePrompt(
          prompt,
          needLlm.map((i) => ({ index: i, text: claims[i] })),
          finalText,
        ),
      });
      inputTokens = res.usage?.inputTokens ?? 0;
      outputTokens = res.usage?.outputTokens ?? 0;
      for (const s of res.object.scores) {
        const i = s.claim_index;
        if (!needLlm.includes(i)) continue;
        scores[i] = s.score;
        scoredBy[i] = "llm";
        evidence[i] = s.evidence;
        explanation[i] = s.explanation;
      }
      const unanswered = needLlm.filter((i) => scores[i] === null);
      if (unanswered.length) {
        judgeError = `judge omitted ${unanswered.length} of ${needLlm.length} claim(s)`;
      }
    } catch (err) {
      // Never silently a fail: an unscorable task is n/a and counted separately.
      judgeError = `judge failed: ${(err as Error).message}`;
    }
  } else if (needLlm.length) {
    judgeError = `${needLlm.length} claim(s) need a judge but no model was supplied`;
  }

  const complete = scores.every((s) => s !== null);
  const coverage =
    complete && scores.length
      ? scores.reduce<number>((a, b) => a + (b ?? 0), 0) / scores.length
      : null;

  const claimScores: ClaimScore[] = screens.map((s, i) => ({
    claim_id: s.claim_id,
    claim: s.claim,
    score: (scores[i] ?? 0) as 0 | 0.5 | 1,
    evidence: evidence[i],
    explanation: explanation[i],
  }));

  const autoScored = scoredBy.filter((x) => x === "screen").length;

  return {
    claims: claimScores,
    coverage,
    verdict: verdictFor(coverage, passThreshold, partialThreshold),
    judge_model: model ? String((model as { modelId?: string }).modelId ?? "unknown") : "none",
    judge_error: judgeError,
    judge_wall_ms: Date.now() - started,
    judge_input_tokens: inputTokens,
    judge_output_tokens: outputTokens,
    scored_by: scoredBy,
    screens,
    claims_auto_scored: autoScored,
    claims_sent_to_llm: needLlm.length,
    auto_rate: screens.length ? autoScored / screens.length : 0,
  };
}

export interface CalibrationRow {
  claim_id: string;
  screen_verdict: ClaimScreen["verdict"];
  screen_score: number | null;
  llm_score: number;
  agrees: boolean;
}

/** Screen-vs-judge agreement on claims the screen was willing to decide.
 *  The screen is only trustworthy to the extent this is measured, so the
 *  calibration path judges everything and reports where they diverge. */
export function calibration(result: McpAtlasRubricResult): {
  rows: CalibrationRow[];
  agreement: number | null;
} {
  const rows: CalibrationRow[] = [];
  for (let i = 0; i < result.screens.length; i++) {
    const s = result.screens[i];
    if (s.verdict === "ambiguous" || result.scored_by[i] !== "llm") continue;
    const llm = result.claims[i].score;
    rows.push({
      claim_id: s.claim_id,
      screen_verdict: s.verdict,
      screen_score: screenScore(s),
      llm_score: llm,
      agrees: screenScore(s) === llm,
    });
  }
  return {
    rows,
    agreement: rows.length ? rows.filter((r) => r.agrees).length / rows.length : null,
  };
}
