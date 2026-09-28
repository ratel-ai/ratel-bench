import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildRunnerModels,
  cacheTier,
  capsLine,
  checkResumeCaps,
  findModelCatalogEntry,
  guardResumeCaps,
  type Harness,
  harnessOf,
  loadModelCatalog,
  type ModelCatalogEntry,
  preferCacheRow,
  resolveOutputCap,
} from "./output-limits.js";
import type { ResolvedModel } from "./types.js";

const QWEN_URL = "https://host.example/prod/v1#qwen3-4b";

const CATALOG: ModelCatalogEntry[] = [
  {
    id: "claude-haiku-4-5",
    bedrockProfile: "eu.anthropic.claude-haiku-4-5",
    maxOutputTokens: 4096,
  },
  { id: "gpt-5.6-luna", maxOutputTokens: 16384 },
  { id: "self-hosted", endpoint: "https://host.example/prod/v1#served", maxOutputTokens: 2048 },
  // Self-hosted, no declared cap → null (no cap sent).
  { id: "qwen3-4b", endpoint: QWEN_URL },
];

describe("resolveOutputCap", () => {
  it("resolves a catalog route by canonical identity or explicit alias", () => {
    const entry: ModelCatalogEntry = {
      id: "bedrock/anthropic.claude-sonnet-5",
      aliases: ["bedrock/claude-sonnet-5", "anthropic/claude-sonnet-5"],
      bedrockProfile: "global.anthropic.claude-sonnet-5",
    };
    expect(findModelCatalogEntry("claude-sonnet-5", [entry])).toBe(entry);
    expect(findModelCatalogEntry("anthropic/claude-sonnet-5", [entry])).toBe(entry);
    expect(findModelCatalogEntry("openai/claude-sonnet-5", [entry])).toBeUndefined();
  });

  it("resolves Vertex Claude aliases without dropping the version suffix", () => {
    const entry: ModelCatalogEntry = {
      id: "gcp/publishers/anthropic/models/claude-sonnet-4-5@20250929",
      aliases: ["gcp/claude-sonnet-4-5@20250929"],
      maxOutputTokens: 8192,
    };
    expect(resolveOutputCap("gcp/claude-sonnet-4-5@20250929", [entry])).toBe(8192);
    expect(findModelCatalogEntry("gcp/claude-sonnet-4-5@20250930", [entry])).toBeUndefined();
  });
  it("uses canonical serving identities and explicit catalog aliases", () => {
    const catalog: ModelCatalogEntry[] = [
      {
        id: "bedrock/anthropic.claude-haiku-4-5",
        aliases: ["claude-haiku-4-5", "anthropic/claude-haiku-4-5"],
        maxOutputTokens: 4096,
      },
      { id: "openai/gpt-5.4-mini", maxOutputTokens: 16384 },
    ];
    expect(resolveOutputCap("anthropic.claude-haiku-4-5", catalog)).toBe(4096);
    expect(resolveOutputCap("anthropic/claude-haiku-4-5", catalog)).toBe(4096);
    expect(resolveOutputCap("openai/gpt-5.4-mini", catalog)).toBe(16384);
    expect(resolveOutputCap("gpt-5.4-mini", catalog)).toBeNull();
  });
  it("override > entry > null", () => {
    expect(resolveOutputCap("claude-haiku-4-5", CATALOG, 1234)).toBe(1234);
    expect(resolveOutputCap("claude-haiku-4-5", CATALOG)).toBe(4096);
    expect(resolveOutputCap("qwen3-4b", CATALOG)).toBeNull();
    expect(resolveOutputCap("ollama:qwen3.5", CATALOG)).toBeNull();
  });

  it("matches a <url>#name id via the entry's endpoint", () => {
    expect(resolveOutputCap("https://host.example/prod/v1#served", CATALOG)).toBe(2048);
    expect(resolveOutputCap(QWEN_URL, CATALOG)).toBeNull();
  });

  it("'none' → null, even when the catalog declares a cap", () => {
    expect(resolveOutputCap("claude-haiku-4-5", CATALOG, "none")).toBeNull();
  });

  it("throws on a 0 / negative / non-integer override or catalog value", () => {
    for (const bad of [0, -1, 1.5, Number.NaN]) {
      expect(() => resolveOutputCap("claude-haiku-4-5", CATALOG, bad)).toThrow(/positive integer/);
      expect(() => resolveOutputCap("m", [{ id: "m", maxOutputTokens: bad as number }])).toThrow(
        /models\.json.*m.*positive integer/,
      );
    }
    expect(() => resolveOutputCap("m", [{ id: "m", maxOutputTokens: "4096" as never }])).toThrow(
      /positive integer/,
    );
  });
});

describe("loadModelCatalog", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "output-limits-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    vi.unstubAllEnvs();
  });

  it("reads run[] entries", () => {
    const path = join(dir, "models.json");
    writeFileSync(path, JSON.stringify({ run: CATALOG }));
    expect(loadModelCatalog(path)).toEqual(CATALOG);
  });

  it("keeps historical overrides addressable outside the campaign default", () => {
    const path = join(dir, "models.json");
    writeFileSync(
      path,
      JSON.stringify({
        run: [{ id: "bedrock/openai.gpt-6-sol", maxOutputTokens: 16 }],
        historical: [{ id: "openai/gpt-5.4-mini", maxOutputTokens: 8 }],
      }),
    );
    expect(loadModelCatalog(path).map(({ id }) => id)).toEqual([
      "bedrock/openai.gpt-6-sol",
      "openai/gpt-5.4-mini",
    ]);
  });

  it("missing catalog → empty (every id resolves to null)", () => {
    const catalog = loadModelCatalog(join(dir, "absent.json"));
    expect(catalog).toEqual([]);
    expect(resolveOutputCap("claude-haiku-4-5", catalog)).toBeNull();
  });

  it("throws on an unparseable catalog instead of silently dropping every cap", () => {
    const path = join(dir, "models.json");
    writeFileSync(path, "{ not json");
    expect(() => loadModelCatalog(path)).toThrow(/models\.json|parse|JSON/i);
  });

  it("rejects malformed and colliding catalog routes", () => {
    const path = join(dir, "models.json");
    writeFileSync(path, JSON.stringify({ run: [{ id: "future/model" }] }));
    expect(() => loadModelCatalog(path)).toThrow(/unknown model provider/);
    writeFileSync(
      path,
      JSON.stringify({ run: [{ id: "bedrock/m" }, { id: "anthropic/m", aliases: ["bedrock/m"] }] }),
    );
    expect(() => loadModelCatalog(path)).toThrow(/duplicate catalog route/);
    writeFileSync(path, JSON.stringify({ run: [{ id: "bedrock/m", aliases: "anthropic/m" }] }));
    expect(() => loadModelCatalog(path)).toThrow(/aliases must be an array/);
    writeFileSync(path, JSON.stringify({ run: [{ id: "bedrock/m" }, { id: "bedrock/m" }] }));
    expect(() => loadModelCatalog(path)).toThrow(/duplicate catalog route/);
  });

  // The CLIs call loadModelCatalog() with no path: a broken default would silently uncap every run.
  it("defaults to the MODELS_JSON overlay (which replaces, not merges, the repo file)", () => {
    const overlay = join(dir, "overlay.json");
    writeFileSync(overlay, JSON.stringify({ run: [{ id: "overlay-only", maxOutputTokens: 777 }] }));
    vi.stubEnv("MODELS_JSON", overlay);
    const catalog = loadModelCatalog();
    expect(resolveOutputCap("overlay-only", catalog)).toBe(777);
    expect(resolveOutputCap("claude-haiku-4-5", catalog)).toBeNull();
  });

  it("defaults to the repo models.json when MODELS_JSON is unset", () => {
    vi.stubEnv("MODELS_JSON", "");
    expect(resolveOutputCap("claude-haiku-4-5", loadModelCatalog())).toBe(4096);
  });
});

describe("buildRunnerModels", () => {
  it("stores canonical model IDs before resolving or writing rows", () => {
    const seen: string[] = [];
    const models = buildRunnerModels(
      ["claude-haiku-4-5", "anthropic/claude-haiku-4-5", "ollama:qwen3.5"],
      (id) => {
        seen.push(id);
        return { id, model: {} as never };
      },
      { catalog: [], override: undefined },
    );
    expect(seen).toEqual([
      "bedrock/claude-haiku-4-5",
      "anthropic/claude-haiku-4-5",
      "ollama:qwen3.5",
    ]);
    expect(models.map((m) => m.id)).toEqual(seen);
  });
  // Mimics the AWS harness's injected Bedrock GUARD: an early `return { id, model }`
  // inside the resolver, with no cap. Caps are attached after resolution, so it can't skip them.
  const guardResolver = (modelId: string): ResolvedModel => ({
    id: modelId,
    model: { provider: "amazon-bedrock", modelId } as never,
  });

  it("attaches caps to a bare {id, model} resolver result", () => {
    const models = buildRunnerModels(["claude-haiku-4-5", "ollama:qwen3.5"], guardResolver, {
      catalog: CATALOG,
      override: undefined,
    });
    expect(models.map((m) => [m.id, m.maxOutputTokens])).toEqual([
      ["bedrock/claude-haiku-4-5", 4096],
      ["ollama:qwen3.5", null],
    ]);
    expect(models[0].model).toEqual({
      provider: "amazon-bedrock",
      modelId: "bedrock/claude-haiku-4-5",
    });
  });

  it("applies the --max-output-tokens override to every model", () => {
    const capped = buildRunnerModels(["claude-haiku-4-5", "ollama:x"], guardResolver, {
      catalog: CATALOG,
      override: 16,
    });
    expect(capped.map((m) => m.maxOutputTokens)).toEqual([16, 16]);
    const none = buildRunnerModels(["claude-haiku-4-5"], guardResolver, {
      catalog: CATALOG,
      override: "none",
    });
    expect(none[0].maxOutputTokens).toBeNull();
  });
});

describe("capsLine", () => {
  it("logs every model's cap and calls out uncapped ones", () => {
    const models = buildRunnerModels(
      ["claude-haiku-4-5", "gpt-5.6-luna", "ollama:qwen3.5"],
      (id) => ({ id, model: {} as never }),
      { catalog: CATALOG, override: undefined },
    );
    expect(capsLine(models)).toBe(
      "caps: bedrock/claude-haiku-4-5=4096, bedrock/gpt-5.6-luna=16384, ollama:qwen3.5=none (no models.json cap)",
    );
    const none = models.map((m) => ({ ...m, maxOutputTokens: null }));
    expect(capsLine(none.slice(0, 1), "none")).toBe(
      "caps: bedrock/claude-haiku-4-5=none (--max-output-tokens none)",
    );
  });
});

describe("harnessOf", () => {
  it("reads the SDK provider off the model and the requested cap", () => {
    expect(
      harnessOf({ model: { provider: "amazon-bedrock" } as never, maxOutputTokens: 4096 }),
    ).toEqual({ provider: "amazon-bedrock", max_output_tokens: 4096 });
  });
});

describe("cacheTier", () => {
  const bedrock = (cap: number | null): Harness => ({
    provider: "amazon-bedrock",
    max_output_tokens: cap,
  });

  it("exact on the same provider and cap (null matches null); never on a different one", () => {
    expect(cacheTier(bedrock(4096), bedrock(4096))).toBe("exact");
    expect(cacheTier(bedrock(null), bedrock(null))).toBe("exact");
    expect(cacheTier(bedrock(16384), bedrock(4096))).toBeNull();
    expect(cacheTier(bedrock(null), bedrock(4096))).toBeNull();
    expect(
      cacheTier({ provider: "anthropic.messages", max_output_tokens: 4096 }, bedrock(4096)),
    ).toBeNull();
  });

  it("a row with no recorded cap is legacy; a recorded provider must still match", () => {
    expect(cacheTier({}, bedrock(4096))).toBe("legacy");
    expect(cacheTier({ provider: "amazon-bedrock" }, bedrock(4096))).toBe("legacy");
    expect(cacheTier({ provider: "anthropic.messages" }, bedrock(4096))).toBeNull();
  });

  it("a same-provider row with no recorded cap is exact for an uncapped run (same request)", () => {
    expect(cacheTier({ provider: "amazon-bedrock" }, bedrock(null))).toBe("exact");
    expect(cacheTier({}, bedrock(null))).toBe("legacy");
  });

  it("without allowLegacy (an explicit --max-output-tokens) only exact rows qualify", () => {
    expect(cacheTier({}, bedrock(16), false)).toBeNull();
    expect(cacheTier({}, bedrock(null), false)).toBeNull();
    expect(cacheTier({ provider: "amazon-bedrock" }, bedrock(16), false)).toBeNull();
    expect(cacheTier({ provider: "amazon-bedrock" }, bedrock(null), false)).toBe("exact");
    expect(cacheTier(bedrock(16), bedrock(16), false)).toBe("exact");
  });

  // The legacy row is EARLIER (and, reused, sometimes LATER), so neither scan order
  // nor earliest-wins can pick the exact row: only the tier can.
  const legacy = { tag: "legacy", generated_at: "2026-06-01T00:00:00.000Z" };
  const legacyLate = { tag: "legacy-late", generated_at: "2026-10-01T00:00:00.000Z" };
  const exact = { tag: "exact", generated_at: "2026-09-01T00:00:00.000Z", ...bedrock(4096) };
  it.each([
    ["legacy→exact", [legacy, exact]],
    ["exact→legacy", [exact, legacy]],
    ["exact→later legacy", [exact, legacyLate]],
    ["later legacy→exact", [legacyLate, exact]],
  ])("preferCacheRow: exact beats legacy in %s scan order", (_order, rows) => {
    let best: (typeof rows)[number] | undefined;
    for (const r of rows) best = preferCacheRow(best, r, bedrock(4096));
    expect(best?.tag).toBe("exact");
  });

  it("preferCacheRow honours allowLegacy", () => {
    expect(preferCacheRow(undefined, { generated_at: "a" }, bedrock(16), false)).toBeUndefined();
    expect(preferCacheRow(undefined, { generated_at: "a" }, bedrock(16))).toEqual({
      generated_at: "a",
    });
  });
});

describe("checkResumeCaps", () => {
  const harness = (cap: number | null): Map<string, Harness> =>
    new Map([["m", { provider: "amazon-bedrock", max_output_tokens: cap }]]);
  const caps = harness(4096);

  it("ignores rows of models not in this run (any cap, null or undefined)", () => {
    expect(
      checkResumeCaps(
        [
          { model: "other", max_output_tokens: 16384 },
          { model: "other", max_output_tokens: null },
          { model: "other" },
        ],
        caps,
      ),
    ).toEqual({ legacy: 0 });
  });

  it("throws on a live row with a different cap, printing uncapped as 'none'", () => {
    expect(() => checkResumeCaps([{ model: "m", max_output_tokens: null }], caps)).toThrow(
      /m: output has none, run uses 4096/,
    );
    expect(() => checkResumeCaps([{ model: "m", max_output_tokens: 4096 }], harness(null))).toThrow(
      /m: output has 4096, run uses none/,
    );
  });

  it("skips infra-error rows (transient/access): unmeasured, excluded from metrics", () => {
    const rows = [
      { model: "m", max_output_tokens: null, error: "x", error_class: "access" as const },
      { model: "m", max_output_tokens: 16, error: "x", error_class: "transient" as const },
      { model: "m", error: "Overloaded" },
    ];
    expect(checkResumeCaps(rows, caps)).toEqual({ legacy: 0 });
  });

  it.each([
    "request",
    "timeout",
    "outcome",
  ] as const)("[guard] still throws on a measured %s-error row at a different cap", (error_class) => {
    expect(() =>
      checkResumeCaps([{ model: "m", max_output_tokens: 16, error: "x", error_class }], caps),
    ).toThrow(/output has 16/);
  });

  it("an infra row does not hide a measured row at another cap", () => {
    const rows = [
      { model: "m", max_output_tokens: null, error: "x", error_class: "access" as const },
      { model: "m", max_output_tokens: 16 },
    ];
    expect(() => checkResumeCaps(rows, caps)).toThrow(/output has 16/);
  });

  it("throws on a reused row with a different defined cap; skips reused legacy rows", () => {
    expect(() =>
      checkResumeCaps([{ model: "m", cache_source: "reused", max_output_tokens: 16 }], caps),
    ).toThrow(/output has 16, run uses 4096/);
    expect(
      checkResumeCaps(
        [
          { model: "m", cache_source: "reused" },
          { model: "m", cache_source: "reused", max_output_tokens: 4096 },
        ],
        caps,
      ),
    ).toEqual({ legacy: 0 });
  });

  describe("under an explicit --max-output-tokens (allowLegacy false)", () => {
    // Served into this output by an earlier catalog-cap invocation of the label.
    const reusedLegacy = { model: "m", cache_source: "reused" as const };

    it.each([16, 4096])("throws on a reused legacy row (cap %s)", (cap) => {
      expect(() => checkResumeCaps([reusedLegacy], harness(cap), false)).toThrow(
        /m: output has legacy \(no recorded cap\), run uses \d+.*exact-tier rows only/,
      );
    });

    it("keeps a reused same-provider pre-cap row under `none` (exact tier: same request)", () => {
      expect(
        checkResumeCaps([{ ...reusedLegacy, provider: "amazon-bedrock" }], harness(null), false),
      ).toEqual({ legacy: 0 });
    });

    it("[guard] live legacy rows still only count (warn), as without the flag", () => {
      expect(checkResumeCaps([{ model: "m" }], harness(16), false)).toEqual({ legacy: 1 });
    });
  });

  it("counts live rows with no recorded cap as legacy", () => {
    expect(checkResumeCaps([{ model: "m" }, { model: "m", cache_source: "live" }], caps)).toEqual({
      legacy: 2,
    });
  });
});

describe("guardResumeCaps", () => {
  const models = [{ id: "m", model: {} as never, maxOutputTokens: 4096 }];

  it("warns once with the legacy count and version", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      guardResumeCaps([{ model: "m" }, { model: "m" }], models, "1.2.3", true);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn).toHaveBeenCalledWith(
        "warn: resume: 2 live row(s) at 1.2.3 predate max_output_tokens (cap unknown); " +
          "resuming over them with this run's caps",
      );
    } finally {
      warn.mockRestore();
    }
  });

  it("throws on a conflicting cap and is silent when nothing is legacy", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(() =>
        guardResumeCaps([{ model: "m", max_output_tokens: 16 }], models, "v", true),
      ).toThrow(/output has 16/);
      guardResumeCaps([{ model: "m", max_output_tokens: 4096 }], models, "v", true);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it("passes allowLegacy through: reused legacy rows are refused under an explicit cap", () => {
    const rows = [{ model: "m", cache_source: "reused" as const }];
    guardResumeCaps(rows, models, "v", true);
    expect(() => guardResumeCaps(rows, models, "v", false)).toThrow(/legacy \(no recorded cap\)/);
  });
});
