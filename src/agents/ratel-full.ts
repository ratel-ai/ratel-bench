import { invokeToolTool, searchToolsTool } from "@ratel-ai/sdk";
import { jsonSchema, tool } from "ai";
import { buildCatalog } from "../catalog.js";
import type { Agent, AgentInstance, ToolSpec, TurnInput } from "../types.js";
import {
  buildBaseAgent,
  buildToolDict,
  normalizeInputSchema,
  sanitizeToolName,
} from "./baseline.js";

const SYSTEM_PROMPT =
  "You have access to a small set of pre-discovered tools that are likely relevant, " +
  "plus a `search_tools` / `invoke_tool` gateway that lets you discover and call any " +
  "tool in the broader catalog if the pre-discovered set doesn't fit.";

/**
 * ratel-full: BM25 pre-discovery + gateway. Builds on the baseline by changing
 * only what's fed into the loop — same AI SDK ToolLoopAgent under the hood.
 *
 * Tool dict is computed per-turn (inside `run()`) because pre-discovery
 * depends on the last user message in the turn's input.
 */
export const ratelFullAgent: Agent = {
  id: "ratel-full",
  init: ({ toolPool, model }) => {
    const catalog = buildCatalog(toolPool);
    const searchExec = searchToolsTool(catalog);
    const invokeExec = invokeToolTool(catalog);
    const topK = 5;

    const run: AgentInstance["run"] = async (input: TurnInput) => {
      const lastUser = lastUserMessage(input.messages);
      const preDiscovered: ToolSpec[] = [];
      for (const hit of catalog.search(lastUser, topK)) {
        const exec = catalog.getExecutable(hit.toolId);
        if (!exec) continue;
        preDiscovered.push({
          id: exec.id,
          name: exec.name,
          description: exec.description,
          inputSchema: exec.inputSchema as Record<string, unknown>,
        });
      }

      const { tools, nameToId } = buildToolDict(preDiscovered);
      // Gateway: register `search_tools` / `invoke_tool` as AI-SDK tools. Their
      // ids are already provider-safe (no sanitization needed) and we keep them
      // out of nameToId so token-trace summarization treats them as gateway.
      for (const exec of [searchExec, invokeExec]) {
        const name = sanitizeToolName(exec.name);
        tools[name] = tool({
          description: exec.description,
          inputSchema: jsonSchema(normalizeInputSchema(exec.inputSchema)),
          execute: exec.execute,
        });
      }

      const base = buildBaseAgent({ model, tools, nameToId, systemPrompt: SYSTEM_PROMPT });
      return base.run(input);
    };

    return { run };
  },
};

function lastUserMessage(messages: { role: string; content: string }[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") return messages[i].content;
  }
  return "";
}
