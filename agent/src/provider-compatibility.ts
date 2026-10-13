import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { REPO_ROOT } from "./paths.js";
import {
  buildSuiteManifest,
  normalizeSuiteRequest,
  type SuiteResult,
  sealSuiteResult,
  suiteCoverageStages,
} from "./suite-contract.js";

type FixtureRoute = {
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

type ProviderCompatibilityFixture = {
  routes: FixtureRoute[];
  expectedResultChecksum: string;
};

const RELEASES = {
  sdk: [{ version: "0.12.0", integrity: "sha512-sdk" }],
  core: [{ version: "0.11.0", checksum: "sha256-core" }],
  compatible: [{ sdk: "0.12.0", core: "0.11.0" }],
};

/** Produce the deterministic public envelope consumed by the website importer contract. */
export function buildProviderCompatibilityResult(): SuiteResult {
  const fixture = loadFixture();
  const manifest = buildSuiteManifest(
    normalizeSuiteRequest({
      schemaVersion: 1,
      runOnlyModels: fixture.routes.map(({ id }) => id),
    }),
    RELEASES,
    {
      benchmarkSha: "a".repeat(40),
      awsSha: "b".repeat(40),
      imageDigest: `sha256:${"c".repeat(64)}`,
      bfclScenarioIds: ["bfcl-provider-matrix"],
      srScenarioIds: ["sr-provider-matrix"],
      corpusHashes: { bfcl: "sha256:bfcl-matrix", sragents: "sha256:sr-matrix" },
      catalogChecksum: "sha256:catalog-matrix",
      pricingChecksum: "sha256:prices-matrix",
      smoke: true,
    },
  );
  const reports = [null, ...fixture.routes.map(({ id }) => id)].flatMap((model) =>
    (["bfcl", "sragents"] as const).flatMap((benchmark) =>
      (["bm25", "dense", "hybrid"] as const).map((retriever) => ({
        model,
        benchmark,
        retriever,
        payload: { fixture: "provider-compatibility" },
      })),
    ),
  );
  return sealSuiteResult({
    schemaVersion: 1,
    runId: "fixture-provider-compatibility",
    status: "completed",
    manifest,
    resolvedModels: fixture.routes.map(({ price: _price, ...route }) => route),
    coverage: {
      expected: manifest.expectedWorkUnitKeys.length,
      completed: manifest.expectedWorkUnitKeys.length,
      failed: 0,
      skipped: 0,
      cancelled: 0,
      completedKeys: manifest.expectedWorkUnitKeys,
      stages: suiteCoverageStages(manifest, manifest.expectedWorkUnitKeys),
    },
    reports,
    errors: [],
    diagnostics: [],
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
      infrastructureIncluded: ["offline synthetic fixture"],
      infrastructureExcluded: ["live provider requests"],
      attempts: { billed: 0, unresolved: 0, usageUnknown: 0, failed: 0, retries: 0, judges: 0 },
      rateSnapshotChecksum: "sha256:prices-matrix",
      rateSnapshot: fixture.routes.map(({ id, provider, sourceRegion, adapter, price }) => ({
        model: id,
        provider,
        api: adapter,
        region: sourceRegion,
        contextTier: "standard",
        ...price,
      })),
      usageByProvider: fixture.routes.map(({ id, provider }) => ({
        model: id,
        provider,
        inputTokens: 0,
        outputTokens: 0,
        cachedInputTokens: 0,
        cacheCreationTokens: 0,
        estimatedUsd: 0,
        costSource: "estimate" as const,
      })),
    },
  } as never);
}

function loadFixture(): ProviderCompatibilityFixture {
  return JSON.parse(
    readFileSync(resolve(REPO_ROOT, "fixtures/suite/provider-compatibility.json"), "utf8"),
  ) as ProviderCompatibilityFixture;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const result = buildProviderCompatibilityResult();
  const expected = loadFixture().expectedResultChecksum;
  if (result.checksumSha256 !== expected) {
    throw new Error(`provider compatibility checksum drift: ${result.checksumSha256}`);
  }
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}
