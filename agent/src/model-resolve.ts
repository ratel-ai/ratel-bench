// Well-known provider endpoints. RATEL_MODELS entries are endpoint links
// (`<baseURL>#<model-name>`); when the host is a provider we know, the model is
// routed through that provider's NATIVE API path (same wire format, auth, and
// friendly row id as the name-addressed form) instead of the generic
// OpenAI-compatible client. Unknown hosts (e.g. the ratel-inference-gateway
// serving qwen3-4b) fall through to the generic custom-endpoint client.
//
// This keeps "everything is a link" as the user-facing model format without
// changing what actually runs for the canonical models.

import { anthropic } from "@ai-sdk/anthropic";
import { openai } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";
import { bedrockModelFromProfile, friendlyBedrockId } from "./bedrock.js";
import type { CustomEndpoint } from "./model-endpoint.js";

export interface ResolvedModel {
  id: string;
  model: LanguageModel;
}

/**
 * Resolve an endpoint whose host is a known provider, or return `null` so the
 * caller falls through to the generic OpenAI-compatible client.
 *
 *   https://bedrock-runtime.<region>.amazonaws.com/…#<profile-id>
 *     → native Bedrock (IAM role auth), id = friendly name
 *   https://api.anthropic.com/…#<model>  → native Anthropic API
 *   https://api.openai.com/…#<model>     → native OpenAI API
 */
export function resolveWellKnownEndpoint(ep: CustomEndpoint): ResolvedModel | null {
  let host: string;
  try {
    host = new URL(ep.baseURL).hostname;
  } catch {
    return null;
  }
  if (host === "api.openai.com") {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error(`model ${ep.modelName} via api.openai.com requires OPENAI_API_KEY`);
    }
    return { id: ep.modelName, model: openai(ep.modelName) };
  }
  if (host === "api.anthropic.com") {
    if (!process.env.ANTHROPIC_API_KEY) {
      throw new Error(`model ${ep.modelName} via api.anthropic.com requires ANTHROPIC_API_KEY`);
    }
    return { id: ep.modelName, model: anthropic(ep.modelName) };
  }
  const bedrockHost = host.match(/^bedrock-runtime\.([a-z0-9-]+)\.amazonaws\.com$/);
  if (bedrockHost) {
    return {
      id: friendlyBedrockId(ep.modelName),
      model: bedrockModelFromProfile(ep.modelName, bedrockHost[1]),
    };
  }
  return null;
}
