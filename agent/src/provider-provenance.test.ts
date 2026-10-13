import { describe, expect, it } from "vitest";
import { resolveModel } from "./model-factory.js";
import { buildRunnerModels, cacheTier, guardResumeCaps, harnessOf } from "./output-limits.js";

describe("route provenance", () => {
  it("retains the effective Vertex location from the actual factory", () => {
    const build = (
      id: string,
      options: Parameters<typeof resolveModel>[1],
      catalog: Parameters<typeof buildRunnerModels>[2]["catalog"] = [],
    ) =>
      buildRunnerModels([id], (modelId) => resolveModel(modelId, { catalog, ...options }), {
        catalog,
        override: undefined,
      })[0];
    const europe = build("gcp/gemini-2.5-pro", {
      env: {
        GOOGLE_VERTEX_PROJECT: "ignored-env-project",
        GOOGLE_VERTEX_LOCATION: "global",
      },
      gcpProject: "offline-project",
      gcpLocation: "europe-west4",
      gcpAccessToken: async () => "offline-token",
    });
    const us = build("gcp/gemini-2.5-pro", {
      env: {},
      gcpProject: "offline-project",
      gcpLocation: "us-central1",
      gcpAccessToken: async () => "offline-token",
    });
    const claudeFromEnv = build("gcp/claude-sonnet-4-5@20250929", {
      env: {
        GOOGLE_VERTEX_PROJECT: "offline-project",
        GOOGLE_VERTEX_LOCATION: "asia-east1",
      },
      gcpAccessToken: async () => "offline-token",
    });
    const catalog = [
      {
        id: "gcp/claude-alias",
        publisher: "Anthropic",
        vertexModelId: "claude-sonnet-4-5@20250929",
        vertexLocation: "europe-west1",
      },
    ];
    const claudeFromCatalog = build(
      "gcp/claude-alias",
      {
        env: {},
        gcpProject: "offline-project",
        gcpLocation: "us-central1",
        gcpAccessToken: async () => "offline-token",
      },
      catalog,
    );

    expect(europe).toMatchObject({
      servingProvider: "gcp",
      publisher: "Google",
      resolvedModel: "gemini-2.5-pro",
      vertexLocation: "europe-west4",
    });
    expect(claudeFromEnv).toMatchObject({
      servingProvider: "gcp",
      publisher: "Anthropic",
      resolvedModel: "claude-sonnet-4-5@20250929",
      vertexLocation: "asia-east1",
    });
    expect(claudeFromCatalog.vertexLocation).toBe("europe-west1");
    expect(cacheTier(harnessOf(europe), harnessOf(us))).toBeNull();
    expect(() =>
      guardResumeCaps([{ model: europe.id, ...harnessOf(europe) }], [us], "test", false),
    ).toThrow(/vertex_location/);
  });

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
