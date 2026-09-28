// Guards the committed models.json (not a fixture): every Bedrock- or API-routed
// model must declare its output cap, so no run silently falls back to the
// provider's (backend-dependent) default.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { canonicalModelId } from "./model-identity.js";
import { loadModelCatalog, type ModelCatalogEntry } from "./output-limits.js";
import { REPO_ROOT } from "./paths.js";

const catalog = loadModelCatalog(resolve(REPO_ROOT, "models.json"));
const campaign = JSON.parse(readFileSync(resolve(REPO_ROOT, "models.json"), "utf8"))
  .run as ModelCatalogEntry[];

/** Bedrock-served, or routed to a vendor API by id prefix (cli.ts:resolveModel). */
function isApiRouted(e: ModelCatalogEntry): boolean {
  return (
    e.bedrockProfile !== undefined ||
    e.id.startsWith("anthropic/") ||
    e.id.startsWith("openai/") ||
    e.id.startsWith("gcp/")
  );
}

describe("models.json", () => {
  it("commits exactly the selected 16 Bedrock routes in campaign order", () => {
    expect(campaign.map((entry) => entry.id)).toEqual([
      "bedrock/openai.gpt-6-astra",
      "bedrock/openai.gpt-6-sol",
      "bedrock/openai.gpt-6-luna",
      "bedrock/anthropic.claude-fable-5-1",
      "bedrock/anthropic.claude-opus-5-5",
      "bedrock/anthropic.claude-sonnet-5",
      "bedrock/anthropic.claude-haiku-4-5",
      "bedrock/moonshotai.kimi-k3",
      "bedrock/qwen.qwen3-coder-next",
      "bedrock/openai.gpt-oss-120b-1:0",
      "bedrock/openai.gpt-oss-20b-1:0",
      "bedrock/google.gemma-4-31b",
      "bedrock/google.gemma-4-26b-a4b",
      "bedrock/mistral.mistral-large-3-675b-instruct",
      "bedrock/meta.llama4-scout-17b-instruct-v1:0",
      "bedrock/nvidia.nemotron-super-3-120b",
    ]);
  });

  it("pins each selected invocation route and supported API", () => {
    expect(
      campaign.map(({ bedrockProfile, bedrockRegion, bedrockEndpoint, bedrockApi }) => [
        bedrockProfile,
        bedrockRegion,
        bedrockEndpoint,
        bedrockApi,
      ]),
    ).toEqual([
      ["global.openai.gpt-6-astra", "eu-central-1", "bedrock-runtime", "responses"],
      ["global.openai.gpt-6-sol", "eu-central-1", "bedrock-runtime", "responses"],
      ["global.openai.gpt-6-luna", "eu-central-1", "bedrock-runtime", "responses"],
      ["global.anthropic.claude-fable-5-1", "eu-central-1", "bedrock-runtime", "converse"],
      ["global.anthropic.claude-opus-5-5", "eu-central-1", "bedrock-runtime", "converse"],
      ["global.anthropic.claude-sonnet-5", "eu-central-1", "bedrock-runtime", "converse"],
      [
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "eu-central-1",
        "bedrock-runtime",
        "converse",
      ],
      ["global.moonshotai.kimi-k3", "eu-central-1", "bedrock-runtime", "converse"],
      ["qwen.qwen3-coder-next", "us-east-1", "bedrock-runtime", "converse"],
      ["openai.gpt-oss-120b-1:0", "eu-central-1", "bedrock-runtime", "converse"],
      ["openai.gpt-oss-20b-1:0", "eu-central-1", "bedrock-runtime", "converse"],
      ["google.gemma-4-31b", "eu-central-1", "bedrock-mantle", "chat"],
      ["google.gemma-4-26b-a4b", "eu-central-1", "bedrock-mantle", "chat"],
      ["mistral.mistral-large-3-675b-instruct", "us-east-1", "bedrock-runtime", "converse"],
      ["us.meta.llama4-scout-17b-instruct-v1:0", "us-east-1", "bedrock-runtime", "converse"],
      ["nvidia.nemotron-super-3-120b", "eu-west-1", "bedrock-runtime", "converse"],
    ]);
  });
  it("every Bedrock/API-routed entry declares a positive-integer maxOutputTokens", () => {
    const routed = catalog.filter(isApiRouted);
    expect(routed.length).toBeGreaterThan(0);
    const missing = routed
      .filter((e) => !Number.isInteger(e.maxOutputTokens) || (e.maxOutputTokens ?? 0) < 1)
      .map((e) => e.id);
    expect(missing).toEqual([]);
  });

  it("uses explicit provider or hosted identities for every committed entry", () => {
    expect(catalog.map((entry) => canonicalModelId(entry.id))).toEqual(
      catalog.map((entry) => entry.id),
    );
  });

  it("pins the reviewed cap per model; self-hosted models send none", () => {
    const caps = Object.fromEntries(catalog.map((e) => [e.id, e.maxOutputTokens ?? null]));
    expect(caps).toMatchObject({
      "bedrock/claude-haiku-4-5": 4096,
      "bedrock/claude-sonnet-4-6": 4096,
      "bedrock/claude-opus-4-5": 4096,
      "bedrock/claude-sonnet-5": 16384,
      "bedrock/claude-opus-4-8": 16384,
      "bedrock/claude-fable-5": 16384,
      "openai/gpt-5.4-mini": 16384,
      "openai/gpt-5.6-luna": 16384,
      // vLLM-style max_model_len is unverified: no cap is sent.
      "https://hj1y208qba.execute-api.eu-central-1.amazonaws.com/prod/v1#qwen3-4b": null,
      "https://hj1y208qba.execute-api.eu-central-1.amazonaws.com/prod/v1#mistral-7b-instruct": null,
    });
  });

  it("keeps optional Vertex publishers and exact API IDs out of the Bedrock campaign", () => {
    expect(campaign).toHaveLength(16);
    expect(catalog.find((entry) => entry.id === "gcp/gemini-2.5-pro")).toMatchObject({
      publisher: "Google",
      vertexModelId: "gemini-2.5-pro",
      vertexLocation: "us-central1",
      aliases: ["gcp/publishers/google/models/gemini-2.5-pro"],
      maxOutputTokens: 16384,
    });
    expect(catalog.find((entry) => entry.id === "gcp/claude-sonnet-4-5@20250929")).toMatchObject({
      publisher: "Anthropic",
      vertexModelId: "claude-sonnet-4-5@20250929",
      aliases: ["gcp/publishers/anthropic/models/claude-sonnet-4-5@20250929"],
    });
  });
});
