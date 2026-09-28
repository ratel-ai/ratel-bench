import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { REPO_ROOT } from "./paths.js";
import type { SuiteResult } from "./suite-contract.js";
import {
  assertAttachmentPreflight,
  attachmentBytes,
  buildSuiteManifest,
  expectedWorkUnitKeys,
  normalizeSuiteRequest,
  resolveStablePair,
  sealSuiteResult,
  suiteCoverageStages,
  suiteResultV1Schema,
  validateSuiteResult,
} from "./suite-contract.js";

type SuiteResultBody = Omit<SuiteResult, "checksumSha256">;

const fixture = (name: string): unknown =>
  JSON.parse(readFileSync(resolve(REPO_ROOT, "fixtures/suite", name), "utf8"));

function fixtureBody(name = "result-partial.json"): Record<string, unknown> {
  const result = structuredClone(fixture(name)) as Record<string, unknown>;
  delete result.checksumSha256;
  return result;
}

function completedFixtureBody(): SuiteResultBody {
  return fixtureBody("result-completed.json") as SuiteResultBody;
}

function first<T>(values: readonly T[]): T {
  const value = values[0];
  if (value === undefined) throw new Error("fixture must contain a row");
  return value;
}

function partialFixtureBody(): SuiteResultBody {
  return fixtureBody() as SuiteResultBody;
}

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
  it("rejects malformed nested manifests at the public sealing boundary", () => {
    const invalid = [
      (body: Record<string, unknown>) => {
        const manifest = body.manifest as Record<string, unknown>;
        (manifest.request as Record<string, unknown>).modelConcurrency = -2;
      },
      (body: Record<string, unknown>) => {
        const manifest = body.manifest as Record<string, unknown>;
        manifest.release = { sdk: {}, core: {} };
        manifest.provenance = { corpusHashes: {} };
      },
      (body: Record<string, unknown>) => {
        (body.manifest as Record<string, unknown>).publishable = "false";
      },
      (body: Record<string, unknown>) => {
        const request = (body.manifest as Record<string, unknown>).request as Record<
          string,
          unknown
        >;
        const models = request.models as string[];
        models.push(models[0]);
      },
    ];
    for (const mutate of invalid) {
      const body = fixtureBody();
      mutate(body);
      expect(() => sealSuiteResult(body as never)).toThrow();
    }
  });

  it("applies the same bounds while constructing manifests before spend", () => {
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
    expect(() =>
      buildSuiteManifest({ ...request, modelConcurrency: -2 }, releases, inputs),
    ).toThrow();
    expect(() =>
      buildSuiteManifest(
        {
          ...request,
          excludedModels: Array.from({ length: 17 }, (_, i) => `bedrock/excluded-${i}`),
        },
        releases,
        inputs,
      ),
    ).toThrow();
    expect(() =>
      buildSuiteManifest(request, releases, { ...inputs, catalogChecksum: "x".repeat(129) }),
    ).toThrow();
  });

  it("rejects foreign, duplicate and inconsistent terminal stages", () => {
    const body = fixtureBody();
    const coverage = body.coverage as Record<string, unknown>;
    coverage.stages = [
      {
        model: "x".repeat(2_000_000),
        benchmark: "bfcl",
        retriever: "bm25",
        status: "completed",
        completed: 999_999,
        expected: 1,
      },
    ];
    expect(() => sealSuiteResult(body as never)).toThrow();

    const honest = fixtureBody();
    const honestCoverage = honest.coverage as Record<string, unknown>;
    const stages = honestCoverage.stages as Array<Record<string, unknown>>;
    stages.push(structuredClone(stages[0]));
    expect(() => sealSuiteResult(honest as never)).toThrow(/stage/i);

    const wrongStatus = fixtureBody();
    const wrongStatusStages = (wrongStatus.coverage as Record<string, unknown>).stages as Array<
      Record<string, unknown>
    >;
    wrongStatusStages[0].status = "completed";
    expect(() => sealSuiteResult(wrongStatus as never)).toThrow(/status/);

    const wrongCount = fixtureBody();
    const wrongCountStages = (wrongCount.coverage as Record<string, unknown>).stages as Array<
      Record<string, unknown>
    >;
    wrongCountStages[0].expected = 1;
    expect(() => sealSuiteResult(wrongCount as never)).toThrow(/counts/);
  });
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

  it("validates the completed synthetic fixture without making it publishable", () => {
    const result = validateSuiteResult(fixture("result-completed.json"));
    expect(result.status).toBe("completed");
    expect(result.manifest.publishable).toBe(false);
    expect(result.coverage.completed).toBe(result.coverage.expected);
    expect(result.coverage.stages.every((stage) => stage.status === "completed")).toBe(true);
    expect(result.reports).toHaveLength(12);
    expect(result.costs.rateSnapshot).toHaveLength(1);
    expect(JSON.stringify(result)).not.toContain("notifyToEmails");
  });

  it.each([
    {
      artifact: "resolved models",
      mutation: "duplicate model row",
      mutate: (body: SuiteResultBody) => {
        body.resolvedModels.push(structuredClone(first(body.resolvedModels)));
      },
    },
    {
      artifact: "resolved models",
      mutation: "omitted model row",
      mutate: (body: SuiteResultBody) => {
        body.resolvedModels = [];
      },
    },
    {
      artifact: "resolved models",
      mutation: "foreign model row",
      mutate: (body: SuiteResultBody) => {
        const model = first(body.resolvedModels);
        model.id = "openai/foreign";
        model.provider = "openai";
      },
    },
    {
      artifact: "resolved models",
      mutation: "provider-prefix mismatch",
      mutate: (body: SuiteResultBody) => {
        first(body.resolvedModels).provider = "wrong-provider";
      },
    },
    {
      artifact: "rate snapshot",
      mutation: "duplicate model row",
      mutate: (body: SuiteResultBody) => {
        body.costs.rateSnapshot.push(structuredClone(first(body.costs.rateSnapshot)));
      },
    },
    {
      artifact: "rate snapshot",
      mutation: "omitted model row",
      mutate: (body: SuiteResultBody) => {
        body.costs.rateSnapshot = [];
      },
    },
    {
      artifact: "rate snapshot",
      mutation: "foreign model row",
      mutate: (body: SuiteResultBody) => {
        const rate = first(body.costs.rateSnapshot);
        rate.model = "openai/foreign";
        rate.provider = "openai";
      },
    },
    {
      artifact: "rate snapshot",
      mutation: "provider-prefix mismatch",
      mutate: (body: SuiteResultBody) => {
        first(body.costs.rateSnapshot).provider = "wrong-provider";
      },
    },
    {
      artifact: "provider usage",
      mutation: "duplicate model row",
      mutate: (body: SuiteResultBody) => {
        body.costs.usageByProvider.push(structuredClone(first(body.costs.usageByProvider)));
      },
    },
    {
      artifact: "provider usage",
      mutation: "omitted model row",
      mutate: (body: SuiteResultBody) => {
        body.costs.usageByProvider = [];
      },
    },
    {
      artifact: "provider usage",
      mutation: "foreign model row",
      mutate: (body: SuiteResultBody) => {
        const usage = first(body.costs.usageByProvider);
        usage.model = "openai/foreign";
        usage.provider = "openai";
      },
    },
    {
      artifact: "provider usage",
      mutation: "provider-prefix mismatch",
      mutate: (body: SuiteResultBody) => {
        first(body.costs.usageByProvider).provider = "wrong-provider";
      },
    },
  ])("rejects completed $artifact with a $mutation", ({ mutate }) => {
    const body = completedFixtureBody();
    mutate(body);
    expect(() => sealSuiteResult(body)).toThrow();
  });

  it.each([
    {
      artifact: "rate snapshot",
      mutation: "duplicate model row",
      mutate: (body: SuiteResultBody, completed: SuiteResultBody) => {
        const row = first(completed.costs.rateSnapshot);
        body.costs.rateSnapshot = [row, structuredClone(row)];
      },
    },
    {
      artifact: "rate snapshot",
      mutation: "foreign model row",
      mutate: (body: SuiteResultBody, completed: SuiteResultBody) => {
        body.costs.rateSnapshot = [
          { ...first(completed.costs.rateSnapshot), model: "openai/foreign", provider: "openai" },
        ];
      },
    },
    {
      artifact: "provider usage",
      mutation: "duplicate model row",
      mutate: (body: SuiteResultBody, completed: SuiteResultBody) => {
        const row = first(completed.costs.usageByProvider);
        body.costs.usageByProvider = [row, structuredClone(row)];
      },
    },
    {
      artifact: "provider usage",
      mutation: "foreign model row",
      mutate: (body: SuiteResultBody, completed: SuiteResultBody) => {
        body.costs.usageByProvider = [
          {
            ...first(completed.costs.usageByProvider),
            model: "openai/foreign",
            provider: "openai",
          },
        ];
      },
    },
  ])("rejects partial $artifact with a $mutation", ({ mutate }) => {
    const body = partialFixtureBody();
    mutate(body, completedFixtureBody());
    expect(() => sealSuiteResult(body)).toThrow();
  });

  it("allows partial results to omit rate and usage rows", () => {
    const body = partialFixtureBody();
    body.costs.rateSnapshot = [];
    body.costs.usageByProvider = [];
    expect(() => sealSuiteResult(body)).not.toThrow();
  });

  it.each([
    {
      condition: "missing current-provider cost",
      mutate: (body: SuiteResultBody) => {
        body.costs.currentProviderUsd = null;
      },
    },
    {
      condition: "provider cost conflicting with budget spend",
      mutate: (body: SuiteResultBody) => {
        body.budget.spentUsd = 123;
        body.budget.remainingUsd = 877;
      },
    },
  ])("rejects complete accounting with $condition", ({ mutate }) => {
    const body = completedFixtureBody();
    mutate(body);
    expect(() => sealSuiteResult(body)).toThrow();
  });

  it("rejects complete accounting with reserved spend", () => {
    const body = completedFixtureBody();
    body.budget.reservedUsd = 100;
    body.budget.remainingUsd = 900;
    expect(() => sealSuiteResult(body)).toThrow(
      /complete budget accounting requires zero reserved spend/,
    );
  });

  it("rejects a completed result with a rate snapshot from different frozen pricing", () => {
    const body = completedFixtureBody();
    body.costs.rateSnapshotChecksum = "sha256:different-prices";
    expect(() => sealSuiteResult(body)).toThrow(
      /completed result rate snapshot checksum differs from frozen pricing checksum/,
    );
  });

  it("checks encoded attachment size before spend and rejects oversized diagnostics", () => {
    const result = fixture("result-partial.json");
    expect(attachmentBytes(result)).toBeLessThan(20_000_000);
    expect(() =>
      validateSuiteResult({ ...(result as object), diagnostics: ["x".repeat(9000)] }),
    ).toThrow();
    const body = fixtureBody();
    body.diagnostics = ["\0".repeat(100)];
    expect(() => sealSuiteResult(body as never)).toThrow(/diagnostics/);
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
    const dimensions = [null, ...maxModels].flatMap((model) =>
      (["bfcl", "sragents"] as const).flatMap((benchmark) =>
        (["bm25", "dense", "hybrid"] as const).map((retriever) => ({
          model,
          benchmark,
          retriever,
        })),
      ),
    );
    const escaped = "\0💥".repeat(40);
    const completed = sealSuiteResult({
      schemaVersion: 1,
      runId: "max-valid-envelope",
      status: "completed",
      manifest,
      resolvedModels: maxModels.map((id) => ({
        id,
        provider: "bedrock",
        publisher: "P".repeat(128),
        invocationId: "i".repeat(256),
        endpoint: "e".repeat(256),
        sourceRegion: "r".repeat(64),
        adapter: "a".repeat(128),
        outputLimit: 16_384,
        capabilityProfile: { escaped: escaped.repeat(20) },
      })),
      coverage: {
        expected: manifest.expectedWorkUnitKeys.length,
        completed: manifest.expectedWorkUnitKeys.length,
        failed: 0,
        skipped: 0,
        cancelled: 0,
        completedKeys: manifest.expectedWorkUnitKeys,
        stages: suiteCoverageStages(manifest, manifest.expectedWorkUnitKeys),
      },
      reports: dimensions.map((dimension) => ({
        ...dimension,
        payload: { escaped: escaped.repeat(17) },
      })),
      errors: [],
      diagnostics: Array.from({ length: dimensions.length * 8 }, () => escaped),
      timing: {
        startedAt: "2026-09-28T00:00:00.000Z",
        endedAt: "2026-09-28T00:00:01.000Z",
        wallMs: 1000,
        workerActiveMs: 900,
        llmActiveMs: 800,
      },
      budget: {
        scope: "model_api",
        ceilingUsd: 1000,
        spentUsd: 0,
        reservedUsd: 0,
        remainingUsd: 1000,
        accounting: "complete",
      },
      costs: {
        currentProviderUsd: 0,
        historicalReusedUsd: 0,
        infrastructureEstimateUsd: 0,
        infrastructureIncluded: [escaped],
        infrastructureExcluded: [escaped],
        attempts: { billed: 0, unresolved: 0, usageUnknown: 0, failed: 0, retries: 0, judges: 0 },
        rateSnapshotChecksum: manifest.provenance.pricingChecksum,
        rateSnapshot: maxModels.map((model) => ({
          model,
          provider: "bedrock",
          api: "converse",
          region: "eu-central-1",
          contextTier: "standard",
          inputPer1M: 1,
          outputPer1M: 1,
          cachedInputPer1M: 1,
          cacheCreationPer1M: 1,
        })),
        usageByProvider: maxModels.map((model) => ({
          model,
          provider: "bedrock",
          inputTokens: 0,
          outputTokens: 0,
          cachedInputTokens: 0,
          cacheCreationTokens: 0,
          estimatedUsd: 0,
          costSource: "estimate",
        })),
      },
    });
    expect(validateSuiteResult(completed).status).toBe("completed");
    expect(attachmentBytes(completed)).toBeLessThan(18_000_000);
  }, 15_000);

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
    expect(
      suiteResultV1Schema.safeParse({ ...leaked, checksumSha256: result.checksumSha256 }).success,
    ).toBe(false);
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
