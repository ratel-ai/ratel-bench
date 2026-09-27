// Metering: wraps an agent run, captures every metric we report on, and emits a
// CellResult ready for JSONL persistence. Structural typing on the result lets
// us swap providers/SDK versions without coupling the meter to a specific shape.

import { createRequire } from "node:module";
import { INVOKE_TOOL_ID, SEARCH_TOOLS_ID } from "@ratel-ai/sdk";
import type { LanguageModel } from "ai";
import { classifyError, type ErrorClass } from "./cell-errors.js";
import type { RetryStats } from "./llm-retry.js";
import type { Arm, CellResult, ProgrammaticVerdict, ToolCall } from "./types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

// Resolve the installed SDK version once. Used as the `ratel_version` row
// dimension and (downstream) cache-key component, so a campaign run is
// "ratel v0.1.5 ran on this corpus" rather than "whatever was on the tree".
export const SDK_VERSION: string = (() => {
  const requirePkg = createRequire(import.meta.url);
  const pkg = requirePkg("@ratel-ai/sdk/package.json") as { version: string };
  return pkg.version;
})();

/** Loose shape of `agent.generate()` output we depend on. Keeps us decoupled from AI SDK internals. */
export interface AgentLikeResult {
  text?: string;
  finishReason?: string;
  steps: AgentStep[];
}

/** One agent-loop step, as found in `AgentLikeResult.steps` and passed to `onStepFinish`. */
export interface AgentStep {
  /** `"length"` means the step stopped on the output-token limit (truncated). */
  finishReason?: string;
  toolCalls?: Array<{ toolName: string; input?: unknown }>;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    cachedInputTokens?: number;
    cacheCreationInputTokens?: number;
    totalTokens?: number;
  };
}

/**
 * Collects each completed step as the loop runs (wire `record` into
 * `ToolLoopAgent({ onStepFinish })`). When a later step throws, `generate()`
 * returns nothing, so `meter` falls back to these steps to keep the usage, cost
 * and truncation the cell already incurred. Their tool calls are never scored.
 */
export class StepRecorder {
  readonly steps: AgentStep[] = [];

  readonly record = (step: AgentStep): void => {
    this.steps.push(step);
  };
}

export interface ModelPrice {
  /** USD per 1M input tokens. */
  inputPer1M: number;
  /** USD per 1M output tokens. */
  outputPer1M: number;
  /** USD per 1M cached input tokens (read). */
  cachedInputPer1M: number;
  /** USD per 1M cache-creation tokens (Anthropic). */
  cacheCreationPer1M: number;
}

export type PricingTable = Record<string, ModelPrice>;

/**
 * Fallback price table — intentionally empty. Real rates live on each model's
 * entry in models.json (backend-keyed) and are loaded at runtime into
 * `RunnerConfig.pricing` by {@link file://./pricing.ts} — see `loadModelPricing`.
 * That keeps a model defined in one place and lets an unpriced model run fine at
 * $0. This empty default only applies when no pricing table is passed at all.
 */
export const DEFAULT_PRICING: PricingTable = {};

export function dollarCost(
  modelId: string,
  tokens: {
    input: number;
    output: number;
    cachedInput: number;
    cacheCreation: number;
  },
  pricing: PricingTable = DEFAULT_PRICING,
): number {
  const price = pricing[modelId];
  // Unknown models (incl. `ollama:*` local runs) intentionally return $0 — the
  // caller can spot a stale price table by cross-referencing raw tokens with
  // expected provider rates. For local runs the $0 is real, not stale.
  if (!price) return 0;
  return (
    (tokens.input * price.inputPer1M +
      tokens.output * price.outputPer1M +
      tokens.cachedInput * price.cachedInputPer1M +
      tokens.cacheCreation * price.cacheCreationPer1M) /
    1_000_000
  );
}

export interface MeterContext {
  scenarioId: string;
  /** Scenario category from the corpus (e.g. `bfcl-simple`); `null`/absent when uncategorized. */
  category?: string | null;
  arm: Arm;
  model: string;
  runIndex: number;
  /** Tools the model directly sees this run (= `ToolBundle.activeToolIds.length`). */
  catalogSize: number;
  /**
   * Universe the BM25 ranked against this run (gold + distractors). `null` for
   * pool-size-agnostic arms (e.g. `control-oracle`), which the runner emits once
   * per scenario regardless of `--pool-sizes`.
   */
  poolSize: number | null;
  seed: number;
  /**
   * Map from AI-SDK function name → canonical tool id for direct (non-gateway)
   * tools. Provider tool-name pattern forces sanitization, so the trace's
   * `toolName` may differ from the canonical id; this map restores it.
   */
  nameToId?: ReadonlyMap<string, string>;
  /** AI SDK provider id of the model (see {@link providerOf}); stamped on the row. */
  provider?: string;
  /**
   * The cell's retry counters (`llm-retry.ts`), read once `generate` settles.
   * Plan-pinned U1/U4 rule: a timeout in a cell that saw any retry classifies
   * `transient`. Retry sleeps are paused out of the deadline, so only failed
   * attempts' own active time is the provider's; even an instant 429 flips it.
   */
  retryStats?: Pick<RetryStats, "retries">;
}

const GATEWAY_NAMES = new Set<string>([SEARCH_TOOLS_ID, INVOKE_TOOL_ID]);

/**
 * Run `generate`, time it, and roll the result into a `CellResult`. Returns the
 * row plus the trace of tool calls so judges can run on the same captured data
 * without re-driving the agent. When `generate` throws, the row's usage and cost
 * come from the steps `recorder` saw complete, so a mid-loop failure keeps its
 * cost; the scored trace (tool calls, turns) stays empty so the cell still fails.
 */
export async function meter(
  ctx: MeterContext,
  generate: () => Promise<AgentLikeResult>,
  pricing: PricingTable = DEFAULT_PRICING,
  recorder?: StepRecorder,
): Promise<{ cell: CellResult; raw: AgentLikeResult | null }> {
  const startedAt = Date.now();
  let raw: AgentLikeResult | null = null;
  let error: string | null = null;
  let errorClass: ErrorClass | undefined;
  try {
    raw = await generate();
  } catch (err) {
    error = (err as Error).message ?? String(err);
    errorClass = classifyError(err, { retries: ctx.retryStats?.retries });
  }
  const wallMs = Date.now() - startedAt;

  // `trace` feeds the judges: empty when generate threw, so an errored cell never
  // scores on its partial trace. `usage` meters what the cell spent either way.
  const trace = summarize(raw, ctx.nameToId);
  const usage = raw || !recorder ? trace : summarize({ steps: recorder.steps }, ctx.nameToId);
  const dollars = dollarCost(
    ctx.model,
    {
      input: usage.inputTokens,
      output: usage.outputTokens,
      cachedInput: usage.cachedInputTokens,
      cacheCreation: usage.cacheCreationTokens,
    },
    pricing,
  );

  const cell: CellResult = {
    scenario_id: ctx.scenarioId,
    category: ctx.category ?? null,
    arm: ctx.arm,
    model: ctx.model,
    provider: ctx.provider,
    run_index: ctx.runIndex,
    ratel_version: SDK_VERSION,
    ratel_ai_core_version: RATEL_AI_CORE_VERSION,
    catalog_size: ctx.catalogSize,
    pool_size: ctx.poolSize,
    seed: ctx.seed,
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    cached_input_tokens: usage.cachedInputTokens,
    cache_creation_tokens: usage.cacheCreationTokens,
    total_tokens: usage.totalTokens,
    tool_calls_total: trace.toolCallsTotal,
    tool_calls_unique: trace.toolCallsUnique,
    gateway_calls: trace.gatewayCalls,
    non_gateway_calls: trace.nonGatewayCalls,
    turns: trace.turns,
    effective_tool_ids: trace.effectiveToolIds,
    programmatic_verdict: "n/a" as ProgrammaticVerdict,
    ast_verdict: "n/a" as ProgrammaticVerdict,
    judge_verdict: "n/a",
    final_text: raw?.text ?? "",
    finish_reason: raw?.finishReason ?? (error ? "error" : "unknown"),
    error,
    error_class: errorClass,
    truncated_steps: usage.truncatedSteps,
    max_step_output_tokens: usage.maxStepOutputTokens,
    wall_ms: wallMs,
    dollar_cost: dollars,
    tool_calls: trace.toolCalls,
  };
  return { cell, raw };
}

interface Summary {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  cacheCreationTokens: number;
  totalTokens: number;
  toolCallsTotal: number;
  toolCallsUnique: number;
  gatewayCalls: number;
  nonGatewayCalls: number;
  turns: number;
  toolCalls: ToolCall[];
  effectiveToolIds: string[];
  /** Steps with `finishReason: "length"` (hit the output-token limit). */
  truncatedSteps: number;
  /** Largest `outputTokens` of any single step. */
  maxStepOutputTokens: number;
}

/**
 * AI SDK provider id of a model (`anthropic.messages`, `amazon-bedrock`, ...);
 * undefined for a plain string id, which the SDK resolves elsewhere.
 */
export function providerOf(model: LanguageModel): string | undefined {
  return typeof model === "string" ? undefined : model.provider;
}

/**
 * Unwrap `invoke_tool` calls into the underlying tool that was actually invoked.
 * `search_tools` is dropped (it's a lookup, not an invocation). Direct tool calls
 * (no gateway) pass through unchanged. This is what the programmatic judge
 * compares against the gold trace.
 */
export function effectiveToolIds(calls: ToolCall[]): string[] {
  const out: string[] = [];
  for (const call of calls) {
    if (call.toolId === SEARCH_TOOLS_ID) continue;
    if (call.toolId === INVOKE_TOOL_ID) {
      const inner = call.args?.toolId;
      if (typeof inner === "string") out.push(inner);
      continue;
    }
    out.push(call.toolId);
  }
  return out;
}

/** A tool call after gateway unwrapping: the canonical tool id + the args. */
export interface EffectiveCall {
  toolId: string;
  args: Record<string, unknown>;
}

/**
 * Like {@link effectiveToolIds} but preserves each call's arguments — the
 * argument-level (AST) judge needs them. `search_tools` is dropped; an
 * `invoke_tool` call is unwrapped to its inner tool id and inner args (the SDK
 * may nest the inner args under `args` or spread them alongside `toolId`, so we
 * handle both). Direct tool calls pass through unchanged.
 */
export function effectiveCalls(calls: ToolCall[]): EffectiveCall[] {
  const out: EffectiveCall[] = [];
  for (const call of calls) {
    if (call.toolId === SEARCH_TOOLS_ID) continue;
    if (call.toolId === INVOKE_TOOL_ID) {
      const inner = call.args?.toolId;
      if (typeof inner !== "string") continue;
      const nested = call.args?.args;
      const args =
        nested && typeof nested === "object" && !Array.isArray(nested)
          ? (nested as Record<string, unknown>)
          : Object.fromEntries(
              Object.entries(call.args ?? {}).filter(([k]) => k !== "toolId" && k !== "args"),
            );
      out.push({ toolId: inner, args });
      continue;
    }
    out.push({ toolId: call.toolId, args: call.args ?? {} });
  }
  return out;
}

export function summarize(
  result: AgentLikeResult | null,
  nameToId?: ReadonlyMap<string, string>,
): Summary {
  if (!result) {
    return {
      inputTokens: 0,
      outputTokens: 0,
      cachedInputTokens: 0,
      cacheCreationTokens: 0,
      totalTokens: 0,
      toolCallsTotal: 0,
      toolCallsUnique: 0,
      gatewayCalls: 0,
      nonGatewayCalls: 0,
      turns: 0,
      toolCalls: [],
      effectiveToolIds: [],
      truncatedSteps: 0,
      maxStepOutputTokens: 0,
    };
  }
  let input = 0;
  let output = 0;
  let cached = 0;
  let cacheCreation = 0;
  let total = 0;
  const calls: ToolCall[] = [];
  let gateway = 0;
  let nonGateway = 0;
  let truncatedSteps = 0;
  let maxStepOutput = 0;
  for (const step of result.steps) {
    if (step.finishReason === "length") truncatedSteps++;
    const u = step.usage;
    if (u) {
      maxStepOutput = Math.max(maxStepOutput, u.outputTokens ?? 0);
      input += u.inputTokens ?? 0;
      output += u.outputTokens ?? 0;
      cached += u.cachedInputTokens ?? 0;
      cacheCreation += u.cacheCreationInputTokens ?? 0;
      total += u.totalTokens ?? 0;
    }
    for (const call of step.toolCalls ?? []) {
      const args =
        typeof call.input === "object" && call.input !== null
          ? (call.input as Record<string, unknown>)
          : {};
      // Map sanitized function names back to canonical ids for direct tools;
      // gateway tools (search_tools / invoke_tool) pass through unchanged.
      const canonical = nameToId?.get(call.toolName) ?? call.toolName;
      calls.push({ toolId: canonical, args });
      if (GATEWAY_NAMES.has(canonical)) gateway++;
      else nonGateway++;
    }
  }
  // Some providers don't surface `totalTokens`; fall back to input + output.
  if (total === 0) total = input + output + cached;
  const unique = new Set(calls.map((c) => c.toolId)).size;
  return {
    inputTokens: input,
    outputTokens: output,
    cachedInputTokens: cached,
    cacheCreationTokens: cacheCreation,
    totalTokens: total,
    toolCallsTotal: calls.length,
    toolCallsUnique: unique,
    gatewayCalls: gateway,
    nonGatewayCalls: nonGateway,
    turns: result.steps.length,
    toolCalls: calls,
    effectiveToolIds: effectiveToolIds(calls),
    truncatedSteps,
    maxStepOutputTokens: maxStepOutput,
  };
}
