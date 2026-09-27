// Guards the committed models.json (not a fixture): every Bedrock- or API-routed
// model must declare its output cap, so no run silently falls back to the
// provider's (backend-dependent) default.

import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { loadModelCatalog, type ModelCatalogEntry } from "./output-limits.js";
import { REPO_ROOT } from "./paths.js";

const catalog = loadModelCatalog(resolve(REPO_ROOT, "models.json"));

/** Bedrock-served, or routed to a vendor API by id prefix (cli.ts:resolveModel). */
function isApiRouted(e: ModelCatalogEntry): boolean {
  return e.bedrockProfile !== undefined || e.id.startsWith("claude-") || e.id.startsWith("gpt-");
}

describe("models.json", () => {
  it("every Bedrock/API-routed entry declares a positive-integer maxOutputTokens", () => {
    const routed = catalog.filter(isApiRouted);
    expect(routed.length).toBeGreaterThan(0);
    const missing = routed
      .filter((e) => !Number.isInteger(e.maxOutputTokens) || (e.maxOutputTokens ?? 0) < 1)
      .map((e) => e.id);
    expect(missing).toEqual([]);
  });

  it("pins the reviewed cap per model; self-hosted models send none", () => {
    const caps = Object.fromEntries(catalog.map((e) => [e.id, e.maxOutputTokens ?? null]));
    expect(caps).toMatchObject({
      "claude-haiku-4-5": 4096,
      "claude-sonnet-4-6": 4096,
      "claude-opus-4-5": 4096,
      "claude-sonnet-5": 16384,
      "claude-opus-4-8": 16384,
      "claude-fable-5": 16384,
      "gpt-5.4-mini": 16384,
      "gpt-5.6-luna": 16384,
      // vLLM-style max_model_len is unverified: no cap is sent.
      "qwen3-4b": null,
      "mistral-7b-instruct": null,
    });
  });
});
