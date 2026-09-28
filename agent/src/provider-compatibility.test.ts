import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { createCampaignBudget, createMemoryBudgetStore } from "./campaign-budget.js";
import { cellKeyString } from "./cell-key.js";
import { resolveModel } from "./model-factory.js";
import { loadModelCatalog, type ModelCatalogEntry } from "./output-limits.js";
import { REPO_ROOT } from "./paths.js";
import { loadModelPricing } from "./pricing.js";
import { buildProviderCompatibilityResult } from "./provider-compatibility.js";
import { validateSuiteResult } from "./suite-contract.js";

type CompatibilityRoute = {
  id: string;
  provider: string;
  publisher: string;
  invocationId: string;
  endpoint: string;
  sourceRegion: string;
  adapter: string;
  outputLimit: number;
  capabilityProfile: Record<string, unknown>;
  price: {
    inputPer1M: number;
    outputPer1M: number;
    cachedInputPer1M: number;
    cacheCreationPer1M: number;
  };
};

type CompatibilityFixture = {
  routes: CompatibilityRoute[];
  expectedResultChecksum: string;
};

const fixture = JSON.parse(
  readFileSync(resolve(REPO_ROOT, "fixtures/suite/provider-compatibility.json"), "utf8"),
) as CompatibilityFixture;
describe("offline provider compatibility matrix", () => {
  it("covers five serving prefixes, both Vertex families and direct xAI", () => {
    expect(new Set(fixture.routes.map(({ provider }) => provider))).toEqual(
      new Set(["bedrock", "anthropic", "openai", "gcp", "xai"]),
    );
    expect(
      fixture.routes.filter(({ provider }) => provider === "gcp").map(({ publisher }) => publisher),
    ).toEqual(["Google", "Anthropic"]);
    expect(fixture.routes.find(({ provider }) => provider === "xai")?.adapter).toBe(
      "xai.responses",
    );
  });

  it("keeps route prices and cache identities provider-scoped", () => {
    const pricing = loadModelPricing();
    expect(pricing["bedrock/claude-opus-4-8"].inputPer1M).toBe(5.5);
    expect(pricing["anthropic/claude-opus-4-8"].inputPer1M).toBe(5);
    const cacheKeys = fixture.routes.map(({ id }) =>
      cellKeyString({
        ratelVersion: "0.12.0",
        scenarioId: "shared-scenario",
        arm: "ratel-full",
        model: id,
        runIndex: 0,
        poolSize: 100,
      }),
    );
    expect(new Set(cacheKeys).size).toBe(fixture.routes.length);
  });

  it("admits priced routes against one campaign budget and rejects an unfunded attempt", async () => {
    const routes = Object.fromEntries(
      fixture.routes.map(({ id, price }) => [
        id,
        { price, maxInputTokens: 100, maxOutputTokens: 50 },
      ]),
    );
    const budget = createCampaignBudget(createMemoryBudgetStore(0.0005), routes);
    budget.preflight(
      fixture.routes.map(({ id, price }) => ({ model: id, price, requestedOutputTokens: 50 })),
    );
    await budget.admit("first", fixture.routes[0].id, 50);
    await budget.settle("first", 4_000_000);
    await expect(budget.admit("second", fixture.routes[1].id, 50)).rejects.toMatchObject({
      status: "budget_limited",
    });
  });

  it("fails explicit unconfigured providers instead of falling back", () => {
    for (const id of [
      "anthropic/claude-sonnet-5",
      "openai/gpt-5.4-mini",
      "gcp/gemini-2.5-pro",
      "gcp/claude-sonnet-4-5@20250929",
      "xai/grok-4-fast",
    ]) {
      expect(() => resolveModel(id, { env: {} })).toThrow(/requires/);
    }
  });

  it("seals an importer-ready result with canonical coverage and provenance", () => {
    const result = buildProviderCompatibilityResult();

    expect(validateSuiteResult(result)).toEqual(result);
    expect(result.manifest.expectedWorkUnitKeys).toHaveLength(96);
    expect(new Set(result.manifest.expectedWorkUnitKeys).size).toBe(96);
    expect(result.checksumSha256).toBe(fixture.expectedResultChecksum);
  });

  it("keeps the committed default at 16 Bedrock routes across selected AWS APIs", () => {
    const catalog = loadModelCatalog(resolve(REPO_ROOT, "models.json"));
    const defaults = JSON.parse(readFileSync(resolve(REPO_ROOT, "models.json"), "utf8"))
      .run as ModelCatalogEntry[];
    expect(defaults).toHaveLength(16);
    expect(defaults.every(({ id }) => id.startsWith("bedrock/"))).toBe(true);
    expect(new Set(defaults.map(({ bedrockApi }) => bedrockApi))).toEqual(
      new Set(["responses", "converse", "chat"]),
    );
    expect(catalog.length).toBeGreaterThan(defaults.length);
  });
});
