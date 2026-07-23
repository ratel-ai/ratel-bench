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
 *  eu-central-1). Extend when a new Claude model joins the benchmark set. */
const BEDROCK_PROFILE_IDS: Record<string, string> = {
  "claude-sonnet-4-6": "eu.anthropic.claude-sonnet-4-6",
  "claude-haiku-4-5": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
};

/** True when claude-* models should route through Bedrock instead of the
 *  Anthropic API. */
export function bedrockEnabled(): boolean {
  return process.env.RATEL_LLM_BACKEND === "bedrock";
}

/** Resolve a friendly claude-* id to a Bedrock-backed LanguageModel. */
export function bedrockModel(modelId: string): LanguageModel {
  const profileId = BEDROCK_PROFILE_IDS[modelId];
  if (!profileId) {
    throw new Error(
      `model ${modelId} has no Bedrock inference-profile mapping ` +
        `(known: ${Object.keys(BEDROCK_PROFILE_IDS).join(", ")}) — ` +
        `add it to BEDROCK_PROFILE_IDS in bedrock.ts or unset RATEL_LLM_BACKEND`,
    );
  }
  const provider = createAmazonBedrock({
    region: process.env.AWS_REGION ?? "eu-central-1",
    credentialProvider: fromNodeProviderChain(),
  });
  return provider(profileId);
}
