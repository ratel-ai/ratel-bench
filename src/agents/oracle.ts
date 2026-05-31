import type { LanguageModel } from "ai";
import type { AgentInstance, Scenario, ToolSpec } from "../types.js";
import { buildBaseAgent, buildToolDict } from "./baseline.js";

/**
 * Oracle: only the tools the scenario actually expects across its turns. The
 * upper bound — there are no distractors at all. Pool-size-agnostic.
 */
export function initOracle(args: {
  scenario: Scenario;
  allTools: ToolSpec[];
  model: LanguageModel;
}): AgentInstance {
  const expectedIds = new Set(args.scenario.turns.map((t) => t.expectedTool));
  const expectedSpecs = args.allTools.filter((t) => expectedIds.has(t.id));
  const { tools, nameToId } = buildToolDict(expectedSpecs);
  return buildBaseAgent({ model: args.model, tools, nameToId });
}
