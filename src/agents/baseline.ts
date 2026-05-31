import { INVOKE_TOOL_ID, SEARCH_TOOLS_ID } from "@ratel-ai/sdk";
import {
  type Tool as AISDKTool,
  jsonSchema,
  type LanguageModel,
  stepCountIs,
  ToolLoopAgent,
  tool,
} from "ai";
import type { Agent, AgentInstance, AgentRunResult, ToolSpec, TurnInput } from "../types.js";

const TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;
const GATEWAY_NAMES = new Set<string>([SEARCH_TOOLS_ID, INVOKE_TOOL_ID]);

export interface BaseAgentConfig {
  model: LanguageModel;
  /** AI SDK tool dict, keyed by sanitized name. */
  tools: Record<string, AISDKTool>;
  /** Map sanitized name → canonical tool id, for unwinding tool-call traces. */
  nameToId: Map<string, string>;
  systemPrompt?: string;
  maxSteps?: number;
  timeoutMs?: number;
}

/**
 * The core AI-SDK loop. Every other agent (oracle, ratel-full, ...) assembles
 * its own tool surface and then calls `buildBaseAgent` with it — the loop and
 * the metering are identical across arms.
 */
export function buildBaseAgent(config: BaseAgentConfig): AgentInstance {
  const agent = new ToolLoopAgent({
    model: config.model,
    tools: config.tools,
    toolChoice: "auto",
    instructions: config.systemPrompt,
    stopWhen: stepCountIs(config.maxSteps ?? 8),
  });
  const timeoutMs = config.timeoutMs ?? 60_000;

  return {
    run: async (input: TurnInput): Promise<AgentRunResult> => {
      const started = Date.now();
      let raw: GenerateResult | null = null;
      let error: string | null = null;
      try {
        raw = (await withTimeout(
          agent.generate({ messages: input.messages }),
          timeoutMs,
        )) as GenerateResult;
      } catch (err) {
        error = (err as Error).message ?? String(err);
      }
      const wallMs = Date.now() - started;
      return summarize(raw, config.nameToId, wallMs, error);
    },
  };
}

/**
 * Turn a ToolSpec[] into an AI-SDK tool dict + name→id map. Reused by every
 * agent that wants to expose direct (non-gateway) tools to the model.
 */
export function buildToolDict(specs: ToolSpec[]): {
  tools: Record<string, AISDKTool>;
  nameToId: Map<string, string>;
} {
  const tools: Record<string, AISDKTool> = {};
  const nameToId = new Map<string, string>();
  for (const spec of specs) {
    const name = sanitizeToolName(spec.id);
    if (tools[name]) {
      throw new Error(`tool name collision after sanitization: "${name}" (id "${spec.id}")`);
    }
    tools[name] = tool({
      description: spec.description,
      inputSchema: jsonSchema(normalizeInputSchema(spec.inputSchema)),
      execute: async () => ({ _stub: "benchmark", toolId: spec.id }),
    });
    nameToId.set(name, spec.id);
  }
  return { tools, nameToId };
}

export function sanitizeToolName(id: string): string {
  if (TOOL_NAME_PATTERN.test(id)) return id;
  const replaced = id.replace(/[^a-zA-Z0-9_-]/g, "_").replace(/^[_-]+|[_-]+$/g, "");
  if (replaced.length === 0) throw new Error(`tool id "${id}" sanitizes to empty`);
  return replaced;
}

export function normalizeInputSchema(schema: unknown): Record<string, unknown> {
  const obj = schema && typeof schema === "object" ? (schema as Record<string, unknown>) : {};
  if (typeof obj.type === "string") return obj;
  return { ...obj, type: "object" };
}

/**
 * The baseline arm: every tool in the pool registered directly with the agent,
 * no gateway, no pre-discovery. This is the "fat-context" floor.
 */
export const baselineAgent: Agent = {
  id: "baseline",
  init: ({ toolPool, model }) => {
    const { tools, nameToId } = buildToolDict(toolPool);
    return buildBaseAgent({ model, tools, nameToId });
  },
};

interface GenerateResult {
  text?: string;
  finishReason?: string;
  steps: Array<{
    toolCalls?: Array<{ toolName: string; input?: unknown }>;
    usage?: {
      inputTokens?: number;
      outputTokens?: number;
      cachedInputTokens?: number;
      cacheCreationInputTokens?: number;
      totalTokens?: number;
    };
  }>;
}

function summarize(
  raw: GenerateResult | null,
  nameToId: Map<string, string>,
  wallMs: number,
  error: string | null,
): AgentRunResult {
  if (!raw) {
    return {
      finalText: "",
      toolCalls: [],
      effectiveToolIds: [],
      tokens: { input: 0, output: 0, cachedInput: 0, cacheCreation: 0, total: 0 },
      wallMs,
      finishReason: error ? "error" : "unknown",
      error,
    };
  }
  let input = 0;
  let output = 0;
  let cached = 0;
  let cacheCreation = 0;
  let total = 0;
  const toolCalls: Array<{ toolId: string; args: unknown }> = [];
  for (const step of raw.steps) {
    const u = step.usage;
    if (u) {
      input += u.inputTokens ?? 0;
      output += u.outputTokens ?? 0;
      cached += u.cachedInputTokens ?? 0;
      cacheCreation += u.cacheCreationInputTokens ?? 0;
      total += u.totalTokens ?? 0;
    }
    for (const call of step.toolCalls ?? []) {
      const canonical = nameToId.get(call.toolName) ?? call.toolName;
      const args = call.input && typeof call.input === "object" ? call.input : {};
      toolCalls.push({ toolId: canonical, args });
    }
  }
  if (total === 0) total = input + output + cached;
  return {
    finalText: raw.text ?? "",
    toolCalls,
    effectiveToolIds: effectiveToolIds(toolCalls),
    tokens: { input, output, cachedInput: cached, cacheCreation, total },
    wallMs,
    finishReason: raw.finishReason ?? (error ? "error" : "unknown"),
    error,
  };
}

function effectiveToolIds(calls: Array<{ toolId: string; args: unknown }>): string[] {
  const out: string[] = [];
  for (const c of calls) {
    if (c.toolId === SEARCH_TOOLS_ID) continue;
    if (c.toolId === INVOKE_TOOL_ID) {
      const inner = (c.args as { toolId?: unknown })?.toolId;
      if (typeof inner === "string") out.push(inner);
      continue;
    }
    out.push(c.toolId);
  }
  return out;
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const id = setTimeout(() => reject(new Error(`run timed out after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(id);
        resolve(v);
      },
      (e) => {
        clearTimeout(id);
        reject(e);
      },
    );
  });
}

// Export GATEWAY_NAMES for any consumer that wants to filter gateway calls separately.
export { GATEWAY_NAMES };
