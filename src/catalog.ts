import { ToolCatalog } from "@ratel-ai/sdk";
import type { ToolSpec } from "./types.js";

export function buildCatalog(pool: ToolSpec[]): ToolCatalog {
  const catalog = new ToolCatalog();
  for (const spec of pool) {
    catalog.register({
      id: spec.id,
      name: spec.name,
      description: spec.description,
      inputSchema: spec.inputSchema ?? {},
      outputSchema: {},
      execute: async () => ({ _stub: "benchmark", toolId: spec.id }),
    });
  }
  return catalog;
}
