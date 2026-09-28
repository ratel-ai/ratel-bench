import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { activeBackend, loadModelPricing } from "./pricing.js";

const CATALOG = {
  run: [
    {
      id: "bedrock/claude-haiku-4-5",
      aliases: ["anthropic/claude-haiku-4-5"],
      bedrockProfile: "eu.anthropic.claude-haiku-4-5",
      pricing: {
        bedrock: {
          inputPer1M: 1.0,
          outputPer1M: 5.0,
          cachedInputPer1M: 0.1,
          cacheCreationPer1M: 1.25,
        },
        anthropic: {
          inputPer1M: 2.0,
          outputPer1M: 9.0,
          cachedInputPer1M: 0.2,
          cacheCreationPer1M: 2.5,
        },
      },
    },
    {
      id: "openai/gpt-5.4-mini",
      pricing: {
        openai: {
          inputPer1M: 0.75,
          outputPer1M: 4.5,
          cachedInputPer1M: 0.075,
          cacheCreationPer1M: 0,
        },
      },
    },
    // Self-hosted, no pricing → absent from the table.
    { id: "https://host/v1#qwen3-4b", endpoint: "https://host/v1#qwen3-4b" },
    // Flat ModelPrice (no backend key) → applies to this route.
    {
      id: "bedrock/claude-flat-test",
      pricing: { inputPer1M: 7, outputPer1M: 8, cachedInputPer1M: 0, cacheCreationPer1M: 0 },
    },
  ],
};

describe("loadModelPricing", () => {
  it("prices canonical routes from the same catalog entry without sharing rates", () => {
    const file = join(tmpdir(), `pricing-routes-${process.pid}.json`);
    try {
      writeFileSync(
        file,
        JSON.stringify({
          run: [
            {
              id: "bedrock/claude-haiku-4-5",
              aliases: ["anthropic/claude-haiku-4-5", "gcp/claude-haiku-4-5"],
              pricing: {
                bedrock: {
                  inputPer1M: 1,
                  outputPer1M: 2,
                  cachedInputPer1M: 0.1,
                  cacheCreationPer1M: 1.25,
                },
                anthropic: {
                  inputPer1M: 3,
                  outputPer1M: 4,
                  cachedInputPer1M: 0.3,
                  cacheCreationPer1M: 3.75,
                },
                gcp: {
                  inputPer1M: 5,
                  outputPer1M: 6,
                  cachedInputPer1M: 0.5,
                  cacheCreationPer1M: 6.25,
                },
              },
            },
          ],
        }),
      );
      const rates = loadModelPricing(file);
      expect(rates["bedrock/claude-haiku-4-5"].inputPer1M).toBe(1);
      expect(rates["anthropic/claude-haiku-4-5"].inputPer1M).toBe(3);
      expect(rates["gcp/claude-haiku-4-5"].inputPer1M).toBe(5);
    } finally {
      rmSync(file, { force: true });
    }
  });
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

  it("keeps direct and Bedrock prices separate", () => {
    const t = loadModelPricing(path);
    expect(t["anthropic/claude-haiku-4-5"].inputPer1M).toBe(2.0);
    expect(t["bedrock/claude-haiku-4-5"].inputPer1M).toBe(1.0);
    expect(t["openai/gpt-5.4-mini"].outputPer1M).toBe(4.5);
  });

  it("keeps historical direct rates after they leave the default roster", () => {
    writeFileSync(
      path,
      JSON.stringify({
        run: [{ id: "bedrock/openai.gpt-6-sol" }],
        historical: [
          {
            id: "openai/gpt-5.4-mini",
            pricing: {
              openai: {
                inputPer1M: 0.75,
                outputPer1M: 4.5,
                cachedInputPer1M: 0.075,
                cacheCreationPer1M: 0,
              },
            },
          },
        ],
      }),
    );
    expect(loadModelPricing(path)["openai/gpt-5.4-mini"].inputPer1M).toBe(0.75);
  });

  it("ignores the legacy global backend setting for explicit routes", () => {
    process.env.RATEL_LLM_BACKEND = "bedrock";
    const t = loadModelPricing(path);
    expect(t["anthropic/claude-haiku-4-5"].inputPer1M).toBe(2.0);
  });

  it("omits unpriced (self-hosted) models", () => {
    const t = loadModelPricing(path);
    expect(t["https://host/v1#qwen3-4b"]).toBeUndefined();
  });

  it("accepts a flat ModelPrice for the entry's own route", () => {
    const t = loadModelPricing(path);
    expect(t["bedrock/claude-flat-test"].inputPer1M).toBe(7);
  });

  it("does not copy an unqualified flat rate onto a different serving route", () => {
    writeFileSync(
      path,
      JSON.stringify({
        run: [
          {
            id: "bedrock/claude-test",
            aliases: ["anthropic/claude-test"],
            pricing: { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 0, cacheCreationPer1M: 0 },
          },
        ],
      }),
    );
    const rates = loadModelPricing(path);
    expect(rates["bedrock/claude-test"].inputPer1M).toBe(1);
    expect(rates["anthropic/claude-test"]).toBeUndefined();
  });

  it("leaves incomplete cache rates unknown rather than treating them as free", () => {
    writeFileSync(
      path,
      JSON.stringify({
        run: [
          {
            id: "gcp/claude-test",
            pricing: { gcp: { inputPer1M: 1, outputPer1M: 2 } },
          },
        ],
      }),
    );
    expect(loadModelPricing(path)["gcp/claude-test"]).toBeUndefined();
  });

  it("returns an empty table (never throws) when the catalog is missing", () => {
    expect(loadModelPricing(join(dir, "nope.json"))).toEqual({});
  });

  it("rejects malformed catalog JSON instead of treating every price as unknown", () => {
    writeFileSync(path, "{broken");
    expect(() => loadModelPricing(path)).toThrow(/not valid JSON/);
  });
});

describe("activeBackend", () => {
  const saved = process.env.RATEL_LLM_BACKEND;
  afterEach(() => {
    if (saved === undefined) delete process.env.RATEL_LLM_BACKEND;
    else process.env.RATEL_LLM_BACKEND = saved;
  });

  it("defaults bare names to Bedrock and leaves hosted routes separate", () => {
    expect(activeBackend("gpt-5.4-mini")).toBe("bedrock");
    expect(activeBackend("openai/gpt-5.4-mini")).toBe("openai");
    expect(activeBackend("ollama:qwen3.5")).toBeNull();
  });

  it("does not let the legacy backend env change an explicit route", () => {
    delete process.env.RATEL_LLM_BACKEND;
    expect(activeBackend("claude-haiku-4-5")).toBe("bedrock");
    process.env.RATEL_LLM_BACKEND = "bedrock";
    expect(activeBackend("claude-haiku-4-5")).toBe("bedrock");
    expect(activeBackend("anthropic/claude-haiku-4-5")).toBe("anthropic");
  });
});
