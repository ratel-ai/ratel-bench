import { describe, expect, it } from "vitest";
import { canonicalModelId, canonicalModelList, parseModelIdentity } from "./model-identity.js";

describe("model identity", () => {
  it("defaults bare names to Bedrock and round-trips all five explicit providers", () => {
    expect(canonicalModelId("gpt-6-sol")).toBe("bedrock/gpt-6-sol");
    for (const provider of ["bedrock", "anthropic", "openai", "gcp", "xai"]) {
      const id = `${provider}/publisher/model/v2:0`;
      expect(parseModelIdentity(id)).toEqual({
        kind: "provider",
        provider,
        model: "publisher/model/v2:0",
        canonicalId: id,
      });
    }
    expect(canonicalModelId("gcp/claude-sonnet-4-5@20250929")).toBe(
      "gcp/claude-sonnet-4-5@20250929",
    );
    expect(canonicalModelId("gcp/publishers/anthropic/models/claude-sonnet-4-5@20250929")).toBe(
      "gcp/publishers/anthropic/models/claude-sonnet-4-5@20250929",
    );
  });

  it("keeps hosted URLs and Ollama tags separate from provider syntax", () => {
    expect(canonicalModelId("https://host.example/v1#org/model/v2")).toBe(
      "https://host.example/v1#org/model/v2",
    );
    expect(canonicalModelId("ollama:qwen3.5:4b")).toBe("ollama:qwen3.5:4b");
    expect(canonicalModelId("bedrock:anthropic.claude-haiku-4-5")).toBe(
      "bedrock/anthropic.claude-haiku-4-5",
    );
    expect(canonicalModelId("anthropic:claude-sonnet-4-6")).toBe("anthropic/claude-sonnet-4-6");
    expect(
      canonicalModelId("arn:aws:bedrock:us-west-2:123456789012:inference-profile/example"),
    ).toBe("bedrock/arn:aws:bedrock:us-west-2:123456789012:inference-profile/example");
  });

  it("deduplicates after canonicalization without reordering selected models", () => {
    expect(canonicalModelList(" gpt-6-sol,openai/gpt-6-sol,bedrock/gpt-6-sol ")).toEqual([
      "bedrock/gpt-6-sol",
      "openai/gpt-6-sol",
    ]);
    expect(() => canonicalModelList("")).toThrow(/invalid model identifier/);
  });

  it("rejects malformed and unimplemented provider IDs before inference", () => {
    for (const id of [
      "",
      " ",
      "/model",
      "gcp/",
      "openai/ ",
      "future/model",
      "future:model",
      "ollama:",
      "https://host/v1",
      "https://#model",
      "bedrock:",
    ]) {
      expect(() => parseModelIdentity(id)).toThrow();
    }
  });
});
