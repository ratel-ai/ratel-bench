import { invokeToolTool, SEARCH_TOOLS_ID, searchToolsTool } from "@ratel-ai/sdk";
import { type Tool as AISDKTool, jsonSchema, type ModelMessage, tool } from "ai";
import { buildCatalog } from "../catalog.js";
import { hashString } from "../pool.js";
import type { Agent, AgentInstance, Message, TurnInput } from "../types.js";
import { buildBaseAgent, normalizeInputSchema, sanitizeToolName } from "./baseline.js";

const SYSTEM_PROMPT =
  "You have a `search_tools` / `invoke_tool` gateway over a large tool catalog. " +
  "Relevant tools for the user's request have already been retrieved via " +
  "`search_tools` and appear in the tool results below. Review them and call the " +
  "best match with `invoke_tool` (passing the tool's arguments under `args`). If " +
  "none of them fit the request, call `search_tools` again with a more specific " +
  "query before answering — prefer acting through a tool over replying in prose.";

// How many candidates pre-discovery injects as the synthetic `search_tools`
// result. Higher K raises the chance the gold tool is present (recall) at the
// cost of a larger injected payload and more distractors to pick wrong from;
// 15 favors recall on the 30–180 pools. The model can still re-search if none
// of the injected candidates fit (see SYSTEM_PROMPT).
const PRE_DISCOVERY_K = 15;

/**
 * ratel-full: BM25 pre-discovery + gateway. The pre-discovered top-K is injected
 * as a synthetic `search_tools` call + tool-result message — byte-for-byte what a
 * real gateway call returns — so the model can't tell pre-discovery from
 * self-discovery, and it must `invoke_tool` a hit to use it (exactly like real
 * discovery). The tool dict is just the two gateway tools, so it's constant
 * across turns/scenarios and the request's tool block caches.
 */
export const ratelFullAgent: Agent = {
  id: "ratel-full",
  init: ({ toolPool, model }) => {
    const catalog = buildCatalog(toolPool);
    const searchExec = searchToolsTool(catalog);
    const invokeExec = invokeToolTool(catalog);

    // Gateway-only tool dict, constant across turns/scenarios → cacheable. Their
    // ids are already provider-safe; kept out of nameToId so token-trace
    // summarization treats them as gateway (unwound by effectiveToolIds).
    const tools: Record<string, AISDKTool> = {};
    for (const exec of [searchExec, invokeExec]) {
      tools[sanitizeToolName(exec.name)] = tool({
        description: exec.description,
        inputSchema: jsonSchema(normalizeInputSchema(exec.inputSchema)),
        execute: exec.execute,
      });
    }
    const base = buildBaseAgent({ model, tools, nameToId: new Map(), systemPrompt: SYSTEM_PROMPT });

    const run: AgentInstance["run"] = async (input: TurnInput) => {
      const lastUser = lastUserMessage(input.messages);
      // Run the real gateway search so the injected result is identical to a
      // model-issued `search_tools` call.
      const searchResult = await searchExec.execute({ query: lastUser, topK: PRE_DISCOVERY_K });
      const callId = `seed-search-${hashString(lastUser)}`;

      const assistantSearch: ModelMessage = {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: callId,
            toolName: SEARCH_TOOLS_ID,
            input: { query: lastUser, topK: PRE_DISCOVERY_K },
          },
        ],
      };
      const toolResult: ModelMessage = {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: callId,
            toolName: SEARCH_TOOLS_ID,
            output: { type: "json", value: searchResult as never },
            // Cache the system+tools+user+discovery prefix so the loop's later
            // `invoke_tool` step re-reads it instead of re-billing the payload.
            providerOptions: { anthropic: { cacheControl: { type: "ephemeral" } } },
          },
        ],
      };

      return base.generate([...(input.messages as ModelMessage[]), assistantSearch, toolResult]);
    };

    return { run };
  },
};

function lastUserMessage(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") return messages[i].content;
  }
  return "";
}
