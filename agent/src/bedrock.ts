// Amazon Bedrock backend for claude-* models, switched on via
// `RATEL_LLM_BACKEND=bedrock` (set on the CodeBuild project; unset locally, so
// laptop runs keep hitting the Anthropic API unchanged).
//
// The RunnerModel id stays the FRIENDLY name (`claude-sonnet-4-6`), not the
// Bedrock profile id — metering.ts DEFAULT_PRICING and every report row are
// keyed by that string, so cost/version comparisons stay coherent across
// backends. Only the wire endpoint differs.
//
// Auth: no API key. In CodeBuild the service role supplies credentials through
// the default AWS credential chain (`fromNodeProviderChain`), which also covers
// local `AWS_PROFILE=…` runs for smoke tests.

import { createAmazonBedrock } from "@ai-sdk/amazon-bedrock";
import { fromNodeProviderChain } from "@aws-sdk/credential-providers";
import type { LanguageModel } from "ai";

/** Friendly model id → EU cross-region inference profile (verified ACTIVE in
 *  eu-central-1). Extend when a new Claude model joins the benchmark set — or,
 *  with no code change, via the BEDROCK_PROFILE_MAP env var (JSON object with
 *  the same shape, merged over this table; set it per run on the CodeBuild
 *  project). Find profile ids with:
 *    aws bedrock list-inference-profiles --region eu-central-1 */
const BEDROCK_PROFILE_IDS: Record<string, string> = {
  "claude-sonnet-4-6": "eu.anthropic.claude-sonnet-4-6",
  "claude-haiku-4-5": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
};

/** Parse the BEDROCK_PROFILE_MAP env extension (empty/{} → no-op, so behavior
 *  without the var is identical to the static table). */
function envProfileMap(): Record<string, string> {
  const raw = process.env.BEDROCK_PROFILE_MAP?.trim();
  if (!raw || raw === "{}") return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new Error(`BEDROCK_PROFILE_MAP is not valid JSON: ${(err as Error).message}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`BEDROCK_PROFILE_MAP must be a JSON object of {"model-id": "profile-id"}`);
  }
  for (const [k, v] of Object.entries(parsed)) {
    if (typeof v !== "string") {
      throw new Error(`BEDROCK_PROFILE_MAP["${k}"] must be a string profile id`);
    }
  }
  return parsed as Record<string, string>;
}

const PROFILE_IDS: Record<string, string> = { ...BEDROCK_PROFILE_IDS, ...envProfileMap() };

/** True when claude-* models should route through Bedrock instead of the
 *  Anthropic API. */
export function bedrockEnabled(): boolean {
  return process.env.RATEL_LLM_BACKEND === "bedrock";
}

/** Resolve a friendly claude-* id to a Bedrock-backed LanguageModel. */
export function bedrockModel(modelId: string): LanguageModel {
  const profileId = PROFILE_IDS[modelId];
  if (!profileId) {
    throw new Error(
      `model ${modelId} has no Bedrock inference-profile mapping ` +
        `(known: ${Object.keys(PROFILE_IDS).join(", ")}) — ` +
        `map it via the BEDROCK_PROFILE_MAP env var (JSON), add it to ` +
        `BEDROCK_PROFILE_IDS in bedrock.ts, or unset RATEL_LLM_BACKEND`,
    );
  }
  return bedrockModelFromProfile(profileId);
}

/** Bedrock-backed LanguageModel straight from an inference-profile id (used by
 *  endpoint-link model entries; no mapping table involved). */
export function bedrockModelFromProfile(profileId: string, region?: string): LanguageModel {
  const provider = createAmazonBedrock({
    region: region ?? process.env.AWS_REGION ?? "eu-central-1",
    credentialProvider: fromNodeProviderChain(),
  });
  return provider(profileId);
}

/** Strip Bedrock profile decoration to the friendly model id, so report rows
 *  and pricing keys stay backend-independent:
 *  `eu.anthropic.claude-haiku-4-5-20251001-v1:0` → `claude-haiku-4-5`. */
export function friendlyBedrockId(profileId: string): string {
  return profileId
    .replace(/^(?:eu|us|apac|global)\.anthropic\./, "")
    .replace(/-v\d+:\d+$/, "")
    .replace(/-\d{8}$/, "");
}
