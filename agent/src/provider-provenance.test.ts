import { describe, expect, it } from "vitest";
import { buildRunnerModels, cacheTier, harnessOf } from "./output-limits.js";

describe("route provenance", () => {
  it("identifies uncatalogued Vertex Gemini and Claude overrides", () => {
    const models = buildRunnerModels(
      ["gcp/gemini-3-pro-preview", "gcp/claude-sonnet-4-5@20250929"],
      (id) => ({ id, model: { provider: "vertex.test" } as never }),
      { catalog: [], override: undefined },
    );
    expect(models.map((model) => [model.publisher, model.resolvedModel])).toEqual([
      ["Google", "gemini-3-pro-preview"],
      ["Anthropic", "claude-sonnet-4-5@20250929"],
    ]);
  });
  it("retains publisher and exact Vertex model behind a GCP alias", () => {
    const models = buildRunnerModels(
      ["gcp/claude-alias"],
      (id) => ({ id, model: { provider: "vertex.anthropic.messages" } as never }),
      {
        catalog: [
          {
            id: "gcp/claude-alias",
            publisher: "Anthropic",
            vertexModelId: "claude-sonnet-4-5@20250929",
            vertexLocation: "europe-west4",
            maxOutputTokens: 1024,
          },
        ],
        override: undefined,
      },
    );
    expect(models[0]).toMatchObject({
      id: "gcp/claude-alias",
      servingProvider: "gcp",
      publisher: "Anthropic",
      resolvedModel: "claude-sonnet-4-5@20250929",
      vertexLocation: "europe-west4",
    });
    expect(
      cacheTier(
        {
          provider: "vertex.anthropic.messages",
          max_output_tokens: 1024,
          resolved_model: "claude-sonnet-4-5@20250929",
          vertex_location: "europe-west4",
        },
        harnessOf(models[0]),
      ),
    ).toBe("exact");
    expect(
      cacheTier(
        {
          provider: "vertex.anthropic.messages",
          max_output_tokens: 1024,
          resolved_model: "claude-sonnet-4-5@20250930",
          vertex_location: "europe-west4",
        },
        harnessOf(models[0]),
      ),
    ).toBeNull();
    expect(
      cacheTier(
        {
          provider: "vertex.anthropic.messages",
          serving_provider: "anthropic",
          max_output_tokens: 1024,
          resolved_model: "claude-sonnet-4-5@20250929",
          vertex_location: "europe-west4",
        },
        harnessOf(models[0]),
      ),
    ).toBeNull();
  });
});
