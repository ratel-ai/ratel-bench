// Helpers shared across every agent file (control + non-control). Each agent
// file is otherwise self-contained — it imports these utilities, builds its
// tool dictionary, runs the agent loop via `meter(...)`, and returns the
// CellResult. Keeping the boilerplate here means the agent files read like
// "this is how you wire up Ratel" instead of being mostly schema-normalization.

import type { ExecutableTool } from "@ratel-ai/sdk";
import { type Tool as AISDKTool, jsonSchema, stepCountIs, ToolLoopAgent, tool } from "ai";
import { FatalProviderError } from "../cell-errors.js";
import { cellRetry } from "../llm-retry.js";
import {
  type AgentLikeResult,
  meter,
  type PricingTable,
  providerOf,
  StepRecorder,
} from "../metering.js";
import type { AgentRunInput, CellResult, ToolSpec } from "../types.js";

const TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

/** Sanitize a tool id into a provider-acceptable function name. */
export function sanitizeToolName(id: string): string {
  if (TOOL_NAME_PATTERN.test(id)) return id;
  const replaced = id.replace(/[^a-zA-Z0-9_-]/g, "_").replace(/^[_-]+|[_-]+$/g, "");
  if (replaced.length === 0) {
    throw new Error(`tool id "${id}" sanitizes to an empty function name`);
  }
  return replaced;
}

/**
 * Bundle the executor with each tool definition. The benchmark corpora ship no
 * canned responses, so the executor returns a fixed stub — what matters is the
 * agent's *selection*, not the response payload (per ADR-0006).
 */
export function toExecutable(spec: ToolSpec): ExecutableTool {
  return {
    id: spec.id,
    name: spec.name,
    description: spec.description,
    inputSchema: spec.input_schema,
    outputSchema: spec.output_schema ?? {},
    execute: async (_args) => ({ _stub: "stubbed for benchmark", toolId: spec.id }),
  };
}

export function toAISDK(exec: ExecutableTool): AISDKTool {
  return tool({
    description: exec.description,
    inputSchema: jsonSchema(normalizeInputSchema(exec.inputSchema)),
    execute: exec.execute,
  });
}

/**
 * MetaTool ships plugin tools with `input_schema: {}` (no parameters declared).
 * Anthropic's API rejects any tool whose input_schema lacks `type: "object"`,
 * so we default the type here at the provider-translation seam. An empty JSON
 * Schema means "anything"; for a function-call signature the practical
 * equivalent is "object with no required properties".
 */
export function normalizeInputSchema(schema: unknown): Record<string, unknown> {
  const obj = schema && typeof schema === "object" ? (schema as Record<string, unknown>) : {};
  if (typeof obj.type === "string") return obj;
  return { ...obj, type: "object" };
}

export interface ToolBundle {
  /** Map of (sanitized) tool name → AI SDK tool, ready for `new ToolLoopAgent({ tools })`. */
  tools: Record<string, AISDKTool>;
  /**
   * Canonical ids of every tool exposed to the agent — both direct tools and
   * gateway tools (`search_tools` / `invoke_tool`). This is the "what did the
   * agent see" count surfaced as the `catalog` column in the report.
   */
  activeToolIds: string[];
  /**
   * Sanitized-name → canonical-id, for direct (non-gateway) tools. Provider APIs
   * require tool names to match `^[a-zA-Z0-9_-]+$`, so ids with dots etc. get
   * rewritten before being handed to the SDK. Metering uses this to map the
   * trace's `toolName` back to the canonical id.
   */
  nameToId: Map<string, string>;
}

/**
 * Register one ToolSpec into the bundle, sanitizing its id to a provider-safe
 * tool name. Two distinct canonical ids can sanitize to the same name (e.g.
 * `solve.quadratic_equation` and `solve_quadratic_equation` both → `solve_quadratic_equation`),
 * which real corpora like BFCL contain. We disambiguate by suffixing (`_2`, `_3`, …)
 * so each id gets a unique name and `nameToId` still maps the trace's `toolName`
 * back to the right canonical id. A repeated identical id (deduped pool safety) is
 * a no-op.
 */
export function registerDirect(spec: ToolSpec, bundle: ToolBundle): void {
  const exec = toExecutable(spec);
  let name = sanitizeToolName(exec.id);
  // Same canonical id already registered under this name → nothing to do.
  if (bundle.nameToId.get(name) === exec.id) return;
  // Name taken by a *different* id (sanitization collision) → find a free suffix.
  if (Object.hasOwn(bundle.tools, name)) {
    let i = 2;
    while (Object.hasOwn(bundle.tools, `${name}_${i}`)) i++;
    name = `${name}_${i}`;
  }
  bundle.tools[name] = toAISDK(exec);
  bundle.nameToId.set(name, exec.id);
  bundle.activeToolIds.push(exec.id);
}

/**
 * Register a gateway tool (`search_tools` / `invoke_tool`) into the bundle.
 * Like `registerDirect` but skips the sanitization+nameToId path (gateway ids
 * are already provider-safe and don't need invoke-trace unwrapping). Pushes
 * onto `activeToolIds` so the catalog count includes gateway tools.
 */
export function registerGateway(exec: ExecutableTool, bundle: ToolBundle): void {
  if (Object.hasOwn(bundle.tools, exec.name)) {
    throw new Error(`gateway tool "${exec.name}" is already registered`);
  }
  bundle.tools[exec.name] = toAISDK(exec);
  bundle.activeToolIds.push(exec.id);
}

/** Build a fresh `ToolBundle` containing every spec in `specs`. */
export function buildToolBundle(specs: ToolSpec[]): ToolBundle {
  const bundle: ToolBundle = {
    tools: {},
    activeToolIds: [],
    nameToId: new Map(),
  };
  for (const spec of specs) {
    registerDirect(spec, bundle);
  }
  return bundle;
}

/** Empty bundle for agents whose tool surface is gateway-only (or otherwise non-spec-driven). */
export function emptyToolBundle(): ToolBundle {
  return { tools: {}, activeToolIds: [], nameToId: new Map() };
}

/**
 * Run the metered agent loop over `bundle.tools` and return a `CellResult` with
 * judging fields left as `n/a` (the runner overlays those). Each agent file
 * builds its own bundle (with whatever Ratel/SDK wiring it wants to demonstrate)
 * and then hands off to this helper for the loop + metering boilerplate, which
 * is identical across arms.
 *
 * Honours the output-cap contract on `AgentRunInput.model`: every step's call
 * requests `maxOutputTokens` (omitted when the cap is `null`), and the row
 * records it as `max_output_tokens`, errored cells included.
 *
 * The model is wrapped per cell by the retry policy (`input.retry`, see
 * `llm-retry.ts`; the SDK's own retries are off) and the loop is aborted after
 * `perRunTimeoutMs` of active time (retry sleeps don't count), with a hard
 * backstop at deadline + grace. Retry counters and the policy land on the row.
 *
 * A fatal provider error (gated/missing model, daily cap) writes no row: it is
 * rethrown as a `FatalProviderError` after metering, carrying the cell's spend
 * (`dollarCost`), so the runner can abort the model and still count the dollars.
 */
export async function runMeteredLoop(
  armId: string,
  input: AgentRunInput,
  bundle: ToolBundle,
): Promise<CellResult> {
  // Records each finished step, so a later step throwing still meters the ones before it.
  const recorder = new StepRecorder();
  const retry = cellRetry(
    input.model.model,
    input.perRunTimeoutMs,
    input.retry,
    `${input.scenario.id} · ${armId} · ${input.model.id} · #${input.runIndex}`,
  );
  const agent = new ToolLoopAgent({
    model: retry.model,
    tools: bundle.tools,
    toolChoice: "auto",
    stopWhen: stepCountIs(input.maxSteps),
    maxOutputTokens: input.model.maxOutputTokens ?? undefined,
    maxRetries: 0, // retries are `withRetry`'s job
    onStepFinish: recorder.record,
  });

  let thrown: unknown;
  const generate = async (): Promise<AgentLikeResult> => {
    try {
      const result = await retry.run(() =>
        agent.generate({ prompt: input.scenario.prompt, abortSignal: retry.signal }),
      );
      return result as unknown as AgentLikeResult;
    } catch (err) {
      thrown = err;
      throw err;
    }
  };

  const { cell } = await meter(
    {
      scenarioId: input.scenario.id,
      category: input.scenario.category ?? null,
      arm: armId,
      model: input.model.id,
      runIndex: input.runIndex,
      catalogSize: bundle.activeToolIds.length,
      poolSize: input.poolSize,
      seed: input.seed,
      nameToId: bundle.nameToId,
      provider: providerOf(input.model.model),
      servingProvider: input.model.servingProvider,
      publisher: input.model.publisher,
      resolvedModel: input.model.resolvedModel,
      vertexLocation: input.model.vertexLocation,
      retryStats: retry.stats,
    },
    generate,
    input.pricing as PricingTable | undefined,
    recorder,
  );
  if (retry.stats.fatal) {
    const fatal = thrown instanceof FatalProviderError ? thrown : new FatalProviderError(thrown);
    fatal.dollarCost = cell.dollar_cost ?? 0;
    fatal.unknownCost = cell.dollar_cost === null;
    throw fatal;
  }
  cell.max_output_tokens = input.model.maxOutputTokens;
  Object.assign(cell, retry.rowFields());
  return cell;
}
