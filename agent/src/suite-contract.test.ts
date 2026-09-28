import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { REPO_ROOT } from "./paths.js";
import {
  assertAttachmentPreflight,
  attachmentBytes,
  buildSuiteManifest,
  expectedWorkUnitKeys,
  normalizeSuiteRequest,
  resolveStablePair,
  sealSuiteResult,
  validateSuiteResult,
} from "./suite-contract.js";

const fixture = (name: string): unknown =>
  JSON.parse(readFileSync(resolve(REPO_ROOT, "fixtures/suite", name), "utf8"));

const releases = {
  sdk: [
    { version: "0.12.0", integrity: "sha512-sdk" },
    { version: "0.13.0-rc.7", integrity: "sha512-rc" },
  ],
  core: [
    { version: "0.11.0", checksum: "sha256-core" },
    { version: "0.12.0-rc.1", checksum: "sha256-rc" },
  ],
  compatible: [{ sdk: "0.12.0", core: "0.11.0" }],
};

describe("suite request contract v1", () => {
  it("uses the committed 16 Bedrock routes, all retrievers and one $1000 campaign budget", () => {
    const request = normalizeSuiteRequest(fixture("request-default.json"));
    expect(request.schemaVersion).toBe(1);
    expect(request.models).toHaveLength(16);
    expect(request.models[0]).toBe("bedrock/openai.gpt-6-astra");
    expect(request.models[15]).toBe("bedrock/nvidia.nemotron-super-3-120b");
    expect(request.modelConcurrency).toBe(1);
    expect(request.campaignBudgetUsd).toBe(1000);
    expect(request.notifyToEmails).toEqual(["dev@ratel.sh"]);
  });

  it("replaces, canonicalizes, deduplicates, then excludes; reports absent exclusions", () => {
    const request = normalizeSuiteRequest({
      schemaVersion: 1,
      runOnlyModels: ["openai.gpt-6-sol", "bedrock/openai.gpt-6-sol", "xai/grok-4"],
      excludeModels: ["bedrock/openai.gpt-6-sol", "bedrock/no-such-model"],
      modelConcurrency: 2,
      campaignBudgetUsd: 25.5,
      notifyToEmails: [" DEV@RATEL.SH ", "a@example.org", "a@example.org"],
    });
    expect(request.models).toEqual(["xai/grok-4"]);
    expect(request.excludedModels).toEqual(["bedrock/openai.gpt-6-sol"]);
    expect(request.unmatchedExclusions).toEqual(["bedrock/no-such-model"]);
    expect(request.notifyToEmails).toEqual(["dev@ratel.sh", "a@example.org"]);
    expect(request.modelConcurrency).toBe(2);
    expect(request.campaignBudgetUsd).toBe(25.5);
    expect(
      normalizeSuiteRequest({
        schemaVersion: 1,
        runOnlyModels: "gcp/gemini-2.5-pro,xai/grok-4",
        excludeModels: "xai/grok-4",
      }).models,
    ).toEqual(["gcp/gemini-2.5-pro"]);
  });

  it("rejects malformed or empty selections, concurrency and nonfinite/nonpositive budgets", () => {
    expect(() => normalizeSuiteRequest({ schemaVersion: 1, runOnlyModels: [] })).toThrow();
    expect(() => normalizeSuiteRequest({ schemaVersion: 1, runOnlyModels: ["openai/"] })).toThrow();
    expect(() =>
      normalizeSuiteRequest({
        schemaVersion: 1,
        excludeModels: ["bedrock/nope"],
        runOnlyModels: ["bedrock/nope"],
      }),
    ).toThrow();
    for (const value of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, "1000"]) {
      expect(() => normalizeSuiteRequest({ schemaVersion: 1, campaignBudgetUsd: value })).toThrow();
    }
    for (const value of [0, 17, 1.5]) {
      expect(() => normalizeSuiteRequest({ schemaVersion: 1, modelConcurrency: value })).toThrow();
    }
    expect(() => normalizeSuiteRequest({ schemaVersion: 1, notifyToEmails: ["nope"] })).toThrow();
  });
});

describe("frozen suite manifest and public result v1", () => {
  it("freezes independently resolved latest stable compatible SDK/core and fixed design", () => {
    expect(resolveStablePair(releases)).toEqual({
      sdk: { version: "0.12.0", integrity: "sha512-sdk" },
      core: { version: "0.11.0", checksum: "sha256-core" },
    });
    const manifest = buildSuiteManifest(normalizeSuiteRequest({ schemaVersion: 1 }), releases, {
      benchmarkSha: "a".repeat(40),
      awsSha: "b".repeat(40),
      imageDigest: `sha256:${"c".repeat(64)}`,
      bfclScenarioIds: ["bfcl-1", "bfcl-2"],
      srScenarioIds: ["sr-1", "sr-2"],
      corpusHashes: { bfcl: "sha256:bfcl", sragents: "sha256:sr" },
      catalogChecksum: "sha256:catalog",
      pricingChecksum: "sha256:prices",
      smoke: true,
    });
    expect(manifest.retrievers).toEqual(["bm25", "dense", "hybrid"]);
    expect(manifest.design).toMatchObject({
      bfclScenarios: 599,
      srScenarios: 600,
      llmPool: 100,
      llmTopK: 5,
      repetitions: 1,
      seed: 42,
    });
    expect(manifest.expectedWorkUnitKeys).toHaveLength(16 * 20 + 72);
    expect(new Set(manifest.expectedWorkUnitKeys).size).toBe(manifest.expectedWorkUnitKeys.length);
    expect(JSON.stringify(manifest)).not.toContain("dev@ratel.sh");
  });

  it("copies mutable request and provenance inputs into the frozen manifest", () => {
    const request = normalizeSuiteRequest({ schemaVersion: 1, runOnlyModels: ["bedrock/example"] });
    const inputs = {
      benchmarkSha: "a".repeat(40),
      awsSha: "b".repeat(40),
      imageDigest: `sha256:${"c".repeat(64)}`,
      bfclScenarioIds: ["bfcl-1"],
      srScenarioIds: ["sr-1"],
      corpusHashes: { bfcl: "sha256:bfcl", sragents: "sha256:sr" },
      catalogChecksum: "sha256:catalog",
      pricingChecksum: "sha256:prices",
      smoke: true,
    };
    const manifest = buildSuiteManifest(request, releases, inputs);
    request.models[0] = "bedrock/changed";
    inputs.corpusHashes.bfcl = "sha256:changed";
    expect(manifest.request.models).toEqual(["bedrock/example"]);
    expect(manifest.provenance.corpusHashes.bfcl).toBe("sha256:bfcl");
  });

  it("keeps controls unique across retrievers and keys distinct by pool, k, model and method", () => {
    const keys = expectedWorkUnitKeys(["bedrock/a", "anthropic/a"], ["b"], ["s"]);
    expect(keys).toHaveLength(2 * 10 + 36);
    expect(new Set(keys).size).toBe(keys.length);
    expect(
      keys.filter((key) => key.startsWith("L|") && key.includes("|control-baseline|")),
    ).toHaveLength(4);
  });

  it("validates shared producer/import fixture and excludes private recipients", () => {
    const result = fixture("result-partial.json");
    expect(validateSuiteResult(result).runId).toBe("fixture-run-1");
    expect(validateSuiteResult(result).costs.rateSnapshot).toEqual([]);
    expect(validateSuiteResult(result).costs.usageByProvider).toEqual([]);
    expect(JSON.stringify(result)).not.toContain("dev@ratel.sh");
    expect(() =>
      validateSuiteResult({ ...(result as object), notifyToEmails: ["secret@example.org"] }),
    ).toThrow();
  });

  it("checks encoded attachment size before spend and rejects oversized diagnostics", () => {
    const result = fixture("result-partial.json");
    expect(attachmentBytes(result)).toBeLessThan(20_000_000);
    expect(() =>
      validateSuiteResult({ ...(result as object), diagnostics: ["x".repeat(9000)] }),
    ).toThrow();
  });

  it("bounds the largest supported full campaign including reports and diagnostics", () => {
    const maxModels = Array.from(
      { length: 16 },
      (_, i) => `bedrock/${"x".repeat(240)}${String(i).padStart(2, "0")}`,
    );
    const manifest = buildSuiteManifest(
      normalizeSuiteRequest({ schemaVersion: 1, runOnlyModels: maxModels }),
      releases,
      {
        benchmarkSha: "a".repeat(40),
        awsSha: "b".repeat(40),
        imageDigest: `sha256:${"c".repeat(64)}`,
        bfclScenarioIds: Array.from(
          { length: 599 },
          (_, i) => `b${String(i).padStart(4, "0")}${"x".repeat(123)}`,
        ),
        srScenarioIds: Array.from(
          { length: 600 },
          (_, i) => `s${String(i).padStart(4, "0")}${"x".repeat(123)}`,
        ),
        srDatasetByScenario: [
          "bigcodebench",
          "champ",
          "logicbench",
          "medcalcbench",
          "theoremqa",
          "toolqa",
        ].flatMap((dataset) => Array(100).fill(dataset)),
        corpusHashes: { bfcl: "sha256:bfcl", sragents: "sha256:sr" },
        catalogChecksum: "sha256:catalog",
        pricingChecksum: "sha256:prices",
      },
    );
    expect(manifest.expectedWorkUnitKeys).toHaveLength(95_920 + 21_582);
    expect(() => assertAttachmentPreflight(manifest)).not.toThrow();
    const groups = 16 * 2 * 3 + 2 * 3;
    const worst = {
      manifest,
      completedKeys: manifest.expectedWorkUnitKeys,
      reports: Array.from({ length: groups }, () => "x".repeat(32_768)),
      diagnostics: Array.from({ length: groups * 8 }, () => "x".repeat(512)),
      configurations: Array.from({ length: 16 }, () => "x".repeat(16_384)),
      rates: Array.from({ length: 16 }, () => "x".repeat(16_384)),
      usage: Array.from({ length: 16 }, () => "x".repeat(4096)),
      errors: Array.from({ length: 128 }, () => "x".repeat(1024)),
      infrastructure: "x".repeat(16_384),
    };
    expect(attachmentBytes(worst)).toBeLessThan(20_000_000);
  });

  it("requires the fixed SR six-dataset stratification for a publishable campaign", () => {
    const inputs = {
      benchmarkSha: "a".repeat(40),
      awsSha: "b".repeat(40),
      imageDigest: `sha256:${"c".repeat(64)}`,
      bfclScenarioIds: Array.from({ length: 599 }, (_, i) => `bfcl-${i}`),
      srScenarioIds: Array.from({ length: 600 }, (_, i) => `sragents-${i}`),
      corpusHashes: { bfcl: "sha256:bfcl", sragents: "sha256:sr" },
      catalogChecksum: "sha256:catalog",
      pricingChecksum: "sha256:prices",
    };
    expect(() =>
      buildSuiteManifest(normalizeSuiteRequest({ schemaVersion: 1 }), releases, inputs),
    ).toThrow(/stratification/);
  });

  it("rejects forged coverage and unbounded public configuration", () => {
    const result = validateSuiteResult(fixture("result-partial.json"));
    const duplicated = {
      ...result,
      coverage: {
        ...result.coverage,
        completedKeys: [result.coverage.completedKeys[0], result.coverage.completedKeys[0]],
        completed: 2,
        skipped: result.coverage.skipped - 1,
      },
    };
    expect(() => sealSuiteResult(duplicated)).toThrow(/completed work-unit keys/);
    const { checksumSha256: _checksum, ...body } = result;
    const oversized = {
      ...body,
      resolvedModels: [
        { ...result.resolvedModels[0], capabilityProfile: { giant: "x".repeat(20_000) } },
      ],
    };
    expect(() => sealSuiteResult(oversized)).toThrow(/configuration/);
  });

  it("rejects private fields nested inside the public manifest", () => {
    const result = validateSuiteResult(fixture("result-partial.json"));
    const { checksumSha256: _checksum, ...body } = result;
    const leaked = {
      ...body,
      manifest: {
        ...result.manifest,
        request: { ...result.manifest.request, secretReferences: { mail: "private" } },
      },
    };
    expect(() => sealSuiteResult(leaked)).toThrow();
  });

  it("rejects inconsistent campaign budget and wall time", () => {
    const result = validateSuiteResult(fixture("result-partial.json"));
    const { checksumSha256: _checksum, ...body } = result;
    expect(() =>
      sealSuiteResult({ ...body, budget: { ...body.budget, remainingUsd: 900 } }),
    ).toThrow(/budget/);
    expect(() => sealSuiteResult({ ...body, timing: { ...body.timing, wallMs: 10 } })).toThrow(
      /wall/,
    );
  });

  it("rejects duplicate or unselected report dimensions", () => {
    const result = validateSuiteResult(fixture("result-partial.json"));
    const { checksumSha256: _checksum, ...body } = result;
    expect(() => sealSuiteResult({ ...body, reports: [...body.reports, body.reports[0]] })).toThrow(
      /report/,
    );
    expect(() =>
      sealSuiteResult({ ...body, reports: [{ ...body.reports[0], model: "openai/other" }] }),
    ).toThrow(/report/);
  });
});
