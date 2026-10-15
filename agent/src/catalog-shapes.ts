import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import type { ExecutableTool, Skill } from "@ratel-ai/sdk";
import type { ToolSpec } from "./types.js";

export type SkillSpec = Pick<Skill, "id" | "name" | "description">;

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

/** Stream the SR universe without retaining skill bodies or other unused fields. */
export async function loadSkillUniverse(path: string): Promise<SkillSpec[]> {
  const universe: SkillSpec[] = [];
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const skill = JSON.parse(line) as SkillSpec;
    universe.push({ id: skill.id, name: skill.name, description: skill.description });
  }
  return universe;
}
