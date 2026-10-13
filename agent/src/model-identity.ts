import { parseCustomEndpoint } from "./model-endpoint.js";

export const SERVING_PROVIDERS = ["bedrock", "anthropic", "openai", "gcp", "xai"] as const;
export type ServingProvider = (typeof SERVING_PROVIDERS)[number];

export type ModelIdentity =
  | { kind: "provider"; provider: ServingProvider; model: string; canonicalId: string }
  | { kind: "hosted"; model: string; canonicalId: string }
  | { kind: "ollama"; model: string; canonicalId: string };

/** Parse the serving route without changing the publisher's model ID. */
export function parseModelIdentity(input: string): ModelIdentity {
  if (!input || input.trim() !== input || /\s/.test(input)) {
    throw new Error(`invalid model identifier: ${JSON.stringify(input)}`);
  }
  const endpoint = parseCustomEndpoint(input);
  if (endpoint) return { kind: "hosted", model: endpoint.modelName, canonicalId: input };
  if (input.startsWith("ollama:")) {
    const model = input.slice("ollama:".length);
    if (!model) throw new Error("ollama model identifier requires a tag");
    return { kind: "ollama", model, canonicalId: input };
  }

  const slash = input.indexOf("/");
  const colon = input.indexOf(":");
  const delimiter = slash >= 0 && (colon < 0 || slash < colon) ? slash : colon;
  let provider: ServingProvider = "bedrock";
  let model = input;
  if (delimiter >= 0) {
    const prefix = input.slice(0, delimiter);
    if (isServingProvider(prefix)) {
      provider = prefix;
      model = input.slice(delimiter + 1);
    } else if (
      delimiter === slash ||
      (delimiter === colon && /^[a-z][a-z0-9_-]*$/i.test(prefix) && prefix !== "arn")
    ) {
      throw new Error(`unknown model provider "${prefix}" in "${input}"`);
    }
  }
  if (!model || model.startsWith("/")) {
    throw new Error(`model identifier "${input}" requires a model name`);
  }
  return { kind: "provider", provider, model, canonicalId: `${provider}/${model}` };
}

export function canonicalModelId(input: string): string {
  return parseModelIdentity(input).canonicalId;
}

/** Canonicalize a comma-separated selection, retaining the first occurrence. */
export function canonicalModelList(raw: string): string[] {
  const models = raw.split(",").map((part) => canonicalModelId(part.trim()));
  return [...new Set(models)];
}

function isServingProvider(value: string): value is ServingProvider {
  return (SERVING_PROVIDERS as readonly string[]).includes(value);
}
