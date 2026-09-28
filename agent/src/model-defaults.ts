import { readFileSync } from "node:fs";
import { canonicalModelId } from "./model-identity.js";
import { modelsJsonPath } from "./pricing.js";

/** The ordered funded campaign roster is the committed models.json run list. */
export const DEFAULT_BFCL_MODELS = campaignModels();
export const DEFAULT_SRAGENTS_MODELS = DEFAULT_BFCL_MODELS;
export const DEFAULT_JUDGE_MODEL = "bedrock/anthropic.claude-sonnet-5";

/** The older run-all command chooses direct APIs from configured direct keys. */
export function runAllProviderModels(keys: { anthropic: boolean; openai: boolean }): string[] {
  const models: string[] = [];
  if (keys.anthropic) models.push("anthropic/claude-sonnet-4-6");
  if (keys.openai) models.push("openai/gpt-5.4-mini");
  return models;
}

function campaignModels(): string[] {
  const catalog = JSON.parse(readFileSync(modelsJsonPath(), "utf8")) as {
    run?: Array<{ id: string }>;
  };
  if (!Array.isArray(catalog.run) || catalog.run.length === 0) {
    throw new Error("models.json needs a nonempty run list");
  }
  return [...new Set(catalog.run.map(({ id }) => canonicalModelId(id)))];
}
