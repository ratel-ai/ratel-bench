import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { activeBackend, loadModelPricing } from "./pricing.js";

const CATALOG = {
  run: [
    {
      id: "claude-haiku-4-5",
      bedrockProfile: "eu.anthropic.claude-haiku-4-5",
      pricing: {
        bedrock: { inputPer1M: 1.0, outputPer1M: 5.0, cachedInputPer1M: 0.1, cacheCreationPer1M: 1.25 },
        anthropic: { inputPer1M: 2.0, outputPer1M: 9.0, cachedInputPer1M: 0.2, cacheCreationPer1M: 2.5 },
      },
    },
    {
      id: "gpt-5.4-mini",
      pricing: { openai: { inputPer1M: 0.75, outputPer1M: 4.5, cachedInputPer1M: 0.075, cacheCreationPer1M: 0 } },
    },
    // Self-hosted, no pricing → absent from the table.
    { id: "qwen3-4b", endpoint: "https://host/v1#qwen3-4b" },
    // Flat ModelPrice (no backend key) → applies to any backend.
    { id: "claude-flat-test", pricing: { inputPer1M: 7, outputPer1M: 8, cachedInputPer1M: 0, cacheCreationPer1M: 0 } },
  ],
};

describe("loadModelPricing", () => {
  let dir: string;
  let path: string;
  const savedBackend = process.env.RATEL_LLM_BACKEND;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "pricing-"));
    path = join(dir, "models.json");
    writeFileSync(path, JSON.stringify(CATALOG));
    delete process.env.RATEL_LLM_BACKEND;
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    if (savedBackend === undefined) delete process.env.RATEL_LLM_BACKEND;
    else process.env.RATEL_LLM_BACKEND = savedBackend;
  });

  it("keys claude-* to the Anthropic rate when no backend is set", () => {
    const t = loadModelPricing(path);
    expect(t["claude-haiku-4-5"].inputPer1M).toBe(2.0); // anthropic block
    expect(t["gpt-5.4-mini"].outputPer1M).toBe(4.5);
  });

  it("keys claude-* to the Bedrock rate under RATEL_LLM_BACKEND=bedrock", () => {
    process.env.RATEL_LLM_BACKEND = "bedrock";
    const t = loadModelPricing(path);
    expect(t["claude-haiku-4-5"].inputPer1M).toBe(1.0); // bedrock block
  });

  it("omits unpriced (self-hosted) models", () => {
    const t = loadModelPricing(path);
    expect(t["qwen3-4b"]).toBeUndefined();
  });

  it("accepts a flat ModelPrice for any backend", () => {
    const t = loadModelPricing(path);
    expect(t["claude-flat-test"].inputPer1M).toBe(7);
  });

  it("returns an empty table (never throws) when the catalog is missing", () => {
    expect(loadModelPricing(join(dir, "nope.json"))).toEqual({});
  });
});

describe("activeBackend", () => {
  const saved = process.env.RATEL_LLM_BACKEND;
  afterEach(() => {
    if (saved === undefined) delete process.env.RATEL_LLM_BACKEND;
    else process.env.RATEL_LLM_BACKEND = saved;
  });

  it("routes gpt-* to openai, ollama/links to null", () => {
    expect(activeBackend("gpt-5.4-mini")).toBe("openai");
    expect(activeBackend("ollama:qwen3.5")).toBeNull();
  });

  it("routes claude-* by RATEL_LLM_BACKEND", () => {
    delete process.env.RATEL_LLM_BACKEND;
    expect(activeBackend("claude-haiku-4-5")).toBe("anthropic");
    process.env.RATEL_LLM_BACKEND = "bedrock";
    expect(activeBackend("claude-haiku-4-5")).toBe("bedrock");
  });
});
