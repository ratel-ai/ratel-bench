import type { ExecutableTool } from "@ratel-ai/sdk";
import type { ToolSpec } from "./types.js";

/** Exact metadata indexed by both BFCL retrieval and its embedding artifact. */
export function toolCatalogMetadata(tool: ToolSpec): Omit<ExecutableTool, "execute"> {
  return {
    id: tool.id,
    name: tool.name,
    description: tool.description,
    inputSchema: tool.input_schema,
    outputSchema: tool.output_schema ?? {},
  };
}

/** Candidate catalogs retrieve tools without invoking them. */
export function candidateTool(tool: ToolSpec): ExecutableTool {
  return { ...toolCatalogMetadata(tool), execute: async () => ({}) };
}
