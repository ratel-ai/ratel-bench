/** Default routes for the BFCL and SR-Agents command-line campaigns. */
export const DEFAULT_BFCL_MODELS = ["bedrock/claude-sonnet-5", "bedrock/claude-haiku-4-5"] as const;
export const DEFAULT_SRAGENTS_MODEL = "bedrock/claude-sonnet-5";
export const DEFAULT_JUDGE_MODEL = "bedrock/claude-sonnet-5";

/** The older run-all command chooses direct APIs from configured direct keys. */
export function runAllProviderModels(keys: { anthropic: boolean; openai: boolean }): string[] {
  const models: string[] = [];
  if (keys.anthropic) models.push("anthropic/claude-sonnet-4-6");
  if (keys.openai) models.push("openai/gpt-5.4-mini");
  return models;
}
