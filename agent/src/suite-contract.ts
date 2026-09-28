/** Public, versioned campaign contracts. No provider calls or private credentials here. */
import { createHash } from "node:crypto";
import { z } from "zod";
import { DEFAULT_BFCL_MODELS } from "./model-defaults.js";
import { canonicalModelId } from "./model-identity.js";

export const SUITE_SCHEMA_VERSION = 1;
export const MAX_SUITE_MODELS = 16;
export const MAX_MODEL_CONCURRENCY = 16;
export const MAX_ATTACHMENT_BYTES = 20_000_000;
export const MAX_REPORT_BYTES = 8_192;
export const MAX_DIAGNOSTICS_PER_GROUP = 8;
export const MAX_DIAGNOSTIC_BYTES = 512;
export const MAX_CONFIGURATION_BYTES = 16_384;
export const MAX_IDENTIFIER_BYTES = 256;
export const MAX_STAGE_WORK_UNITS = 4_000;
export const RETRIEVERS = ["bm25", "dense", "hybrid"] as const;
export const ARMS = ["control-baseline", "control-oracle", "ratel-full"] as const;
export const SR_DATASETS = [
  "bigcodebench",
  "champ",
  "logicbench",
  "medcalcbench",
  "theoremqa",
  "toolqa",
] as const;

const modelList = z.union([z.array(z.string()), z.string()]);
export const suiteRequestV1Schema = z
  .object({
    schemaVersion: z.literal(1),
    runOnlyModels: modelList.optional(),
    excludeModels: modelList.optional(),
    modelConcurrency: z.number().int().min(1).max(MAX_MODEL_CONCURRENCY).optional(),
    campaignBudgetUsd: z.number().positive().finite().optional(),
    notifyToEmails: z.array(z.string()).optional(),
  })
  .strict();

const releaseSchema = z
  .object({
    version: z.string(),
    integrity: z.string().optional(),
    checksum: z.string().optional(),
  })
  .strict();
const releaseMetadataSchema = z
  .object({
    sdk: z.array(releaseSchema),
    core: z.array(releaseSchema),
    compatible: z.array(z.object({ sdk: z.string(), core: z.string() }).strict()),
  })
  .strict();

const boundedText = (max: number) =>
  z
    .string()
    .min(1)
    .refine(
      (value) => jsonBytes(value) <= max + 2,
      `serialized string content exceeds ${max} byte budget`,
    );
const canonicalModelSchema = boundedText(MAX_IDENTIFIER_BYTES).refine((value) => {
  try {
    return canonicalModelId(value) === value;
  } catch {
    return false;
  }
}, "canonical provider-qualified model required");
const uniqueModelsSchema = z
  .array(canonicalModelSchema)
  .min(1)
  .max(MAX_SUITE_MODELS)
  .refine((values) => new Set(values).size === values.length, "duplicate model identities");
const boundedModelListSchema = z
  .array(canonicalModelSchema)
  .max(MAX_SUITE_MODELS)
  .refine((values) => new Set(values).size === values.length, "duplicate model identities");
const frozenRequestSchema = z
  .object({
    models: uniqueModelsSchema,
    excludedModels: boundedModelListSchema,
    unmatchedExclusions: boundedModelListSchema,
    modelConcurrency: z.number().int().min(1).max(MAX_MODEL_CONCURRENCY),
    campaignBudgetUsd: z.number().positive().finite(),
  })
  .strict();
const stableReleaseSchema = z
  .object({ version: z.string().regex(/^\d+\.\d+\.\d+$/), integrity: boundedText(256) })
  .strict();
const stableCoreSchema = z
  .object({ version: z.string().regex(/^\d+\.\d+\.\d+$/), checksum: boundedText(256) })
  .strict();
const fixedDesignSchema = z
  .object({
    bfclScenarios: z.literal(599),
    srScenarios: z.literal(600),
    srScenariosPerDataset: z.literal(100),
    srDatasets: z.literal(6),
    srDatasetNames: z.tuple([
      z.literal("bigcodebench"),
      z.literal("champ"),
      z.literal("logicbench"),
      z.literal("medcalcbench"),
      z.literal("theoremqa"),
      z.literal("toolqa"),
    ]),
    seed: z.literal(42),
    repetitions: z.literal(1),
    llmPool: z.literal(100),
    llmTopK: z.literal(5),
    retrievalPools: z
      .object({
        bfcl: z.tuple([z.literal(30), z.literal(100)]),
        sragents: z.tuple([z.literal(50), z.literal(100)]),
      })
      .strict(),
    retrievalTopK: z.tuple([z.literal(1), z.literal(3), z.literal(5)]),
    arms: z.tuple([
      z.literal("control-baseline"),
      z.literal("control-oracle"),
      z.literal("ratel-full"),
    ]),
  })
  .strict();
const suiteManifestV1Schema = z
  .object({
    schemaVersion: z.literal(1),
    publishable: z.boolean(),
    request: frozenRequestSchema,
    release: z.object({ sdk: stableReleaseSchema, core: stableCoreSchema }).strict(),
    retrievers: z.tuple([z.literal("bm25"), z.literal("dense"), z.literal("hybrid")]),
    design: fixedDesignSchema,
    scenarioIds: z
      .object({
        bfcl: z
          .array(z.string().regex(/^[a-zA-Z0-9_.:-]{1,128}$/))
          .min(1)
          .max(599),
        sragents: z
          .array(z.string().regex(/^[a-zA-Z0-9_.:-]{1,128}$/))
          .min(1)
          .max(600),
      })
      .strict(),
    srDatasetByScenario: z.array(z.enum(SR_DATASETS)).max(600),
    expectedWorkUnitKeys: z.array(boundedText(128)).min(1).max(120_000),
    provenance: z
      .object({
        benchmarkSha: z.string().regex(/^[0-9a-f]{40}$/),
        awsSha: z.string().regex(/^[0-9a-f]{40}$/),
        imageDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
        corpusHashes: z.object({ bfcl: boundedText(128), sragents: boundedText(128) }).strict(),
        catalogChecksum: boundedText(128),
        pricingChecksum: boundedText(128),
      })
      .strict(),
  })
  .strict();
const suiteInputsSchema = z
  .object({
    benchmarkSha: z.string().regex(/^[0-9a-f]{40}$/),
    awsSha: z.string().regex(/^[0-9a-f]{40}$/),
    imageDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
    bfclScenarioIds: z.array(z.string()).min(1).max(599),
    srScenarioIds: z.array(z.string()).min(1).max(600),
    srDatasetByScenario: z.array(z.string()).max(600).optional(),
    corpusHashes: z.object({ bfcl: boundedText(128), sragents: boundedText(128) }).strict(),
    catalogChecksum: boundedText(128),
    pricingChecksum: boundedText(128),
    smoke: z.boolean().optional(),
  })
  .strict();

export type Retriever = (typeof RETRIEVERS)[number];
export type Arm = (typeof ARMS)[number];
export type SuiteStatus = "completed" | "partial" | "failed" | "cancelled" | "budget_limited";
export type NormalizedSuiteRequest = {
  schemaVersion: 1;
  models: string[];
  excludedModels: string[];
  unmatchedExclusions: string[];
  modelConcurrency: number;
  campaignBudgetUsd: number;
  notifyToEmails: string[];
};
export type ReleaseMetadata = z.infer<typeof releaseMetadataSchema>;
export type StablePair = {
  sdk: { version: string; integrity: string };
  core: { version: string; checksum: string };
};
export type SuiteInputs = {
  benchmarkSha: string;
  awsSha: string;
  imageDigest: string;
  bfclScenarioIds: string[];
  srScenarioIds: string[];
  srDatasetByScenario?: string[];
  corpusHashes: { bfcl: string; sragents: string };
  catalogChecksum: string;
  pricingChecksum: string;
  smoke?: boolean;
};
export type SuiteManifest = ReturnType<typeof buildSuiteManifest>;

/** Parse CLI/config/Action inputs before any provider or budget admission. */
export function normalizeSuiteRequest(raw: unknown): NormalizedSuiteRequest {
  const input = suiteRequestV1Schema.parse(raw);
  const selected =
    input.runOnlyModels === undefined ? [...DEFAULT_BFCL_MODELS] : parseModels(input.runOnlyModels);
  if (selected.length === 0) throw new Error("runOnlyModels cannot be empty");
  const excludes = input.excludeModels === undefined ? [] : parseModels(input.excludeModels);
  const excludeSet = new Set(excludes);
  const models = selected.filter((model) => !excludeSet.has(model));
  if (models.length === 0) throw new Error("model selection is empty after exclusions");
  if (models.length > MAX_SUITE_MODELS)
    throw new Error(`suite supports at most ${MAX_SUITE_MODELS} models`);
  if (models.some((model) => Buffer.byteLength(model) > 256))
    throw new Error("model identifier exceeds 256 bytes");
  const selectedSet = new Set(selected);
  const notifyToEmails = ["dev@ratel.sh"];
  for (const value of input.notifyToEmails ?? []) {
    const email = value.trim().toLowerCase();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || email.length > 254) {
      throw new Error(`invalid notification email: ${JSON.stringify(value)}`);
    }
    if (!notifyToEmails.includes(email)) notifyToEmails.push(email);
  }
  if (notifyToEmails.length > 32) throw new Error("too many notification recipients");
  return {
    schemaVersion: 1,
    models,
    excludedModels: excludes.filter((model) => selectedSet.has(model)),
    unmatchedExclusions: excludes.filter((model) => !selectedSet.has(model)),
    modelConcurrency: input.modelConcurrency ?? 1,
    campaignBudgetUsd: input.campaignBudgetUsd ?? 1000,
    notifyToEmails,
  };
}

/** Select the highest stable tested pair from submission-time registry metadata. */
export function resolveStablePair(raw: unknown): StablePair {
  const metadata = releaseMetadataSchema.parse(raw);
  const pair = metadata.compatible
    .filter(({ sdk, core }) => stableVersion(sdk) && stableVersion(core))
    .sort((a, b) => compareVersions(b.sdk, a.sdk) || compareVersions(b.core, a.core))
    .find(
      ({ sdk, core }) =>
        metadata.sdk.some((release) => release.version === sdk && release.integrity) &&
        metadata.core.some((release) => release.version === core && release.checksum),
    );
  if (!pair) throw new Error("no stable compatible SDK/core pair with integrity metadata");
  const sdk = metadata.sdk.find((release) => release.version === pair.sdk);
  const core = metadata.core.find((release) => release.version === pair.core);
  if (!sdk?.integrity || !core?.checksum) throw new Error("release integrity missing");
  return {
    sdk: { version: pair.sdk, integrity: sdk.integrity },
    core: { version: pair.core, checksum: core.checksum },
  };
}

/** Freeze the exact suite plan; recipients remain only in the private request. */
export function buildSuiteManifest(
  request: NormalizedSuiteRequest,
  releases: ReleaseMetadata,
  inputs: SuiteInputs,
) {
  const parsedInputs = suiteInputsSchema.parse(inputs);
  const frozenRequest = frozenRequestSchema.parse({
    models: request.models,
    excludedModels: request.excludedModels,
    unmatchedExclusions: request.unmatchedExclusions,
    modelConcurrency: request.modelConcurrency,
    campaignBudgetUsd: request.campaignBudgetUsd,
  });
  inputs = parsedInputs;
  validateScenarioIds(inputs.bfclScenarioIds, 599, Boolean(inputs.smoke), "BFCL");
  validateScenarioIds(inputs.srScenarioIds, 600, Boolean(inputs.smoke), "SR-Agents");
  validateSrStratification(
    inputs.srDatasetByScenario,
    inputs.srScenarioIds.length,
    Boolean(inputs.smoke),
  );
  if (
    jsonBytes(inputs.corpusHashes) > 512 ||
    [inputs.catalogChecksum, inputs.pricingChecksum].some((value) => Buffer.byteLength(value) > 128)
  )
    throw new Error("provenance configuration exceeds cap");
  const expectedWorkUnitKeys = expectedWorkUnitKeysFor(
    frozenRequest.models,
    inputs.bfclScenarioIds,
    inputs.srScenarioIds,
  );
  const manifest = {
    schemaVersion: 1 as const,
    publishable: !inputs.smoke,
    request: {
      models: [...frozenRequest.models],
      excludedModels: [...frozenRequest.excludedModels],
      unmatchedExclusions: [...frozenRequest.unmatchedExclusions],
      modelConcurrency: frozenRequest.modelConcurrency,
      campaignBudgetUsd: frozenRequest.campaignBudgetUsd,
    },
    release: resolveStablePair(releases),
    retrievers: [...RETRIEVERS],
    design: fixedDesign(),
    scenarioIds: { bfcl: [...inputs.bfclScenarioIds], sragents: [...inputs.srScenarioIds] },
    srDatasetByScenario: [...(inputs.srDatasetByScenario ?? [])],
    expectedWorkUnitKeys,
    provenance: {
      benchmarkSha: inputs.benchmarkSha,
      awsSha: inputs.awsSha,
      imageDigest: inputs.imageDigest,
      corpusHashes: { ...inputs.corpusHashes },
      catalogChecksum: inputs.catalogChecksum,
      pricingChecksum: inputs.pricingChecksum,
    },
  };
  assertAttachmentPreflight(manifest);
  return manifest;
}

/** Compact semantic keys: indices address frozen model/scenario dictionaries in the manifest. */
export function expectedWorkUnitKeys(
  models: string[],
  bfclScenarioIds: string[],
  srScenarioIds: string[],
): string[] {
  return expectedWorkUnitKeysFor(models, bfclScenarioIds, srScenarioIds);
}

const money = z.number().finite().nonnegative();
const nullableMoney = money.nullable();
const coverageStatus = z.enum(["completed", "failed", "skipped", "cancelled"]);
const benchmarkSchema = z.enum(["bfcl", "sragents"]);
const coverageStageSchema = z
  .object({
    model: canonicalModelSchema.nullable(),
    benchmark: benchmarkSchema,
    retriever: z.enum(RETRIEVERS),
    status: coverageStatus,
    completed: z.number().int().nonnegative().max(MAX_STAGE_WORK_UNITS),
    expected: z.number().int().positive().max(MAX_STAGE_WORK_UNITS),
  })
  .strict();
export type SuiteCoverageStage = z.infer<typeof coverageStageSchema>;
const reportSchema = z
  .object({
    model: canonicalModelSchema.nullable(),
    benchmark: benchmarkSchema,
    retriever: z.enum(RETRIEVERS),
    payload: z.record(z.unknown()),
  })
  .strict();
export const suiteResultV1Schema = z
  .object({
    schemaVersion: z.literal(1),
    runId: boundedText(128),
    status: z.enum(["completed", "partial", "failed", "cancelled", "budget_limited"]),
    manifest: z.custom<SuiteManifest>((value) => {
      try {
        validateManifest(value as SuiteManifest);
        return true;
      } catch {
        return false;
      }
    }),
    resolvedModels: z.array(
      z
        .object({
          id: canonicalModelSchema,
          provider: boundedText(64),
          publisher: boundedText(128),
          invocationId: boundedText(256),
          endpoint: boundedText(256),
          sourceRegion: boundedText(64),
          adapter: boundedText(128),
          outputLimit: z.number().int().positive(),
          capabilityProfile: z.record(z.unknown()),
        })
        .strict(),
    ),
    coverage: z
      .object({
        expected: z.number().int().nonnegative(),
        completed: z.number().int().nonnegative(),
        failed: z.number().int().nonnegative(),
        skipped: z.number().int().nonnegative(),
        cancelled: z.number().int().nonnegative(),
        completedKeys: z.array(z.string()),
        stages: z.array(coverageStageSchema),
      })
      .strict(),
    reports: z.array(reportSchema),
    errors: z.array(
      z
        .object({
          stage: boundedText(256),
          model: canonicalModelSchema.nullable(),
          message: z.string().max(1024),
        })
        .strict(),
    ),
    diagnostics: z.array(z.string().max(MAX_DIAGNOSTIC_BYTES)),
    timing: z
      .object({
        startedAt: z.string().datetime(),
        endedAt: z.string().datetime(),
        wallMs: z.number().int().nonnegative(),
        workerActiveMs: z.number().int().nonnegative(),
        llmActiveMs: z.number().int().nonnegative(),
      })
      .strict(),
    budget: z
      .object({
        scope: z.literal("model_api"),
        ceilingUsd: money,
        spentUsd: nullableMoney,
        reservedUsd: money,
        remainingUsd: nullableMoney,
        accounting: z.enum(["complete", "partial", "unknown"]),
      })
      .strict(),
    costs: z
      .object({
        currentProviderUsd: nullableMoney,
        historicalReusedUsd: nullableMoney,
        infrastructureEstimateUsd: nullableMoney,
        infrastructureIncluded: z.array(boundedText(512)).max(32),
        infrastructureExcluded: z.array(boundedText(512)).max(32),
        attempts: z
          .object({
            billed: z.number().int().nonnegative(),
            unresolved: z.number().int().nonnegative(),
            usageUnknown: z.number().int().nonnegative(),
            failed: z.number().int().nonnegative(),
            retries: z.number().int().nonnegative(),
            judges: z.number().int().nonnegative(),
          })
          .strict(),
        rateSnapshotChecksum: boundedText(128),
        rateSnapshot: z.array(
          z
            .object({
              model: canonicalModelSchema,
              provider: boundedText(64),
              api: boundedText(64),
              region: boundedText(64),
              contextTier: boundedText(64),
              inputPer1M: money,
              outputPer1M: money,
              cachedInputPer1M: money,
              cacheCreationPer1M: money,
            })
            .strict(),
        ),
        usageByProvider: z.array(
          z
            .object({
              model: canonicalModelSchema,
              provider: boundedText(64),
              inputTokens: z.number().int().nonnegative().nullable(),
              outputTokens: z.number().int().nonnegative().nullable(),
              cachedInputTokens: z.number().int().nonnegative().nullable(),
              cacheCreationTokens: z.number().int().nonnegative().nullable(),
              estimatedUsd: nullableMoney,
              costSource: z.enum(["provider", "estimate", "partial", "unknown"]),
            })
            .strict(),
        ),
      })
      .strict(),
    checksumSha256: z.string().regex(/^[0-9a-f]{64}$/),
  })
  .strict();
export type SuiteResult = z.infer<typeof suiteResultV1Schema>;

/** Validate the exact public envelope consumed by finalizer, mailer and website importer. */
export function validateSuiteResult(raw: unknown): SuiteResult {
  const result = suiteResultV1Schema.parse(raw);
  const manifest = result.manifest;
  validateManifest(manifest);
  if (JSON.stringify(manifest).includes("notifyToEmails"))
    throw new Error("private recipients in public manifest");
  if (
    result.resolvedModels.length !== manifest.request.models.length ||
    result.resolvedModels.some((model, index) => model.id !== manifest.request.models[index])
  )
    throw new Error("resolved models do not match request");
  if (result.resolvedModels.some((model) => jsonBytes(model) > MAX_CONFIGURATION_BYTES))
    throw new Error("model capability configuration exceeds cap");
  if (result.coverage.expected !== manifest.expectedWorkUnitKeys.length)
    throw new Error("expected work-unit count mismatch");
  const expected = new Set(manifest.expectedWorkUnitKeys);
  if (expected.size !== manifest.expectedWorkUnitKeys.length)
    throw new Error("duplicate expected work-unit keys");
  const completed = new Set(result.coverage.completedKeys);
  if (
    completed.size !== result.coverage.completedKeys.length ||
    [...completed].some((key) => !expected.has(key))
  )
    throw new Error("invalid completed work-unit keys");
  if (
    result.coverage.completed !== completed.size ||
    result.coverage.completed +
      result.coverage.failed +
      result.coverage.skipped +
      result.coverage.cancelled !==
      result.coverage.expected
  )
    throw new Error("coverage does not reconcile");
  validateCoverageStages(manifest, result.coverage, completed);
  if (
    result.status === "completed" &&
    (result.coverage.completed !== result.coverage.expected ||
      result.budget.accounting !== "complete")
  )
    throw new Error("completed result lacks publishable coverage/accounting");
  if (result.budget.ceilingUsd !== manifest.request.campaignBudgetUsd)
    throw new Error("campaign budget differs from frozen request");
  if (result.budget.accounting === "complete") {
    if (
      result.budget.spentUsd === null ||
      result.budget.remainingUsd === null ||
      result.costs.attempts.unresolved > 0 ||
      result.costs.attempts.usageUnknown > 0
    )
      throw new Error("complete budget accounting has unresolved spend");
    if (
      Math.abs(
        result.budget.ceilingUsd -
          result.budget.spentUsd -
          result.budget.reservedUsd -
          result.budget.remainingUsd,
      ) > 0.000001
    )
      throw new Error("campaign budget does not reconcile");
  }
  const elapsed = Date.parse(result.timing.endedAt) - Date.parse(result.timing.startedAt);
  if (elapsed < 0 || elapsed !== result.timing.wallMs)
    throw new Error("campaign wall time does not reconcile");
  if (
    result.costs.rateSnapshot.length > manifest.request.models.length ||
    result.costs.usageByProvider.length > manifest.request.models.length
  )
    throw new Error("cost snapshot exceeds selected models");
  if (
    result.status === "completed" &&
    result.costs.rateSnapshot.length !== manifest.request.models.length
  )
    throw new Error("publishable result needs selected-model price snapshots");
  if (
    result.costs.rateSnapshot.some((rate) => jsonBytes(rate) > MAX_CONFIGURATION_BYTES) ||
    result.costs.usageByProvider.some((usage) => jsonBytes(usage) > 4096)
  )
    throw new Error("cost configuration exceeds cap");
  if (
    jsonBytes([result.costs.infrastructureIncluded, result.costs.infrastructureExcluded]) > 16_384
  )
    throw new Error("infrastructure estimate details exceed cap");
  for (const report of result.reports)
    if (jsonBytes(report.payload) > MAX_REPORT_BYTES) throw new Error("report payload exceeds cap");
  const reportDimensions = new Set<string>();
  for (const report of result.reports) {
    if (report.model !== null && !manifest.request.models.includes(report.model))
      throw new Error("report model was not selected");
    const dimension = JSON.stringify([report.model, report.benchmark, report.retriever]);
    if (reportDimensions.has(dimension)) throw new Error("duplicate report dimensions");
    reportDimensions.add(dimension);
  }
  if (
    result.status === "completed" &&
    reportDimensions.size !== (manifest.request.models.length + 1) * 2 * RETRIEVERS.length
  )
    throw new Error("completed result lacks report payloads");
  if (result.reports.length > (manifest.request.models.length + 1) * 2 * RETRIEVERS.length)
    throw new Error("too many reports");
  if (result.coverage.stages.length > (manifest.request.models.length + 1) * 2 * RETRIEVERS.length)
    throw new Error("too many coverage stages");
  if (result.errors.length > 128 || result.errors.some((error) => jsonBytes(error) > 1100))
    throw new Error("errors exceed diagnostic cap");
  if (
    result.diagnostics.length >
      MAX_DIAGNOSTICS_PER_GROUP * (manifest.request.models.length + 1) * 2 * RETRIEVERS.length ||
    result.diagnostics.some((value) => jsonBytes(value) > MAX_DIAGNOSTIC_BYTES)
  )
    throw new Error("diagnostics exceed cap");
  const { checksumSha256: _checksum, ...body } = result;
  if (checksum(body) !== result.checksumSha256) throw new Error("result checksum mismatch");
  if (attachmentBytes(result) > MAX_ATTACHMENT_BYTES)
    throw new Error("result exceeds encoded attachment budget");
  return result;
}

/** Finalizer computes checksum before validating and writing the immutable result. */
export function sealSuiteResult(body: Omit<SuiteResult, "checksumSha256">): SuiteResult {
  return validateSuiteResult({ ...body, checksumSha256: checksum(body) });
}

/** Base64 payload size; includes a conservative 1 KiB MIME/header allowance. */
export function attachmentBytes(value: unknown): number {
  return 4 * Math.ceil(jsonBytes(value) / 3) + 1024;
}

/** Reject unsupported cardinality/configuration before model spend. */
export function assertAttachmentPreflight(manifest: SuiteManifest): void {
  validateManifest(manifest);
  const reportSlots =
    manifest.request.models.length * 2 * RETRIEVERS.length + 2 * RETRIEVERS.length;
  const diagnostics = reportSlots * MAX_DIAGNOSTICS_PER_GROUP * (MAX_DIAGNOSTIC_BYTES + 4);
  const reports = reportSlots * (MAX_REPORT_BYTES + 1024);
  const completedKeys = jsonBytes(manifest.expectedWorkUnitKeys);
  const configurations = manifest.request.models.length * (2 * MAX_CONFIGURATION_BYTES + 4096);
  const errors = 128 * (1024 + 100);
  const stages = reportSlots * (MAX_IDENTIFIER_BYTES + 192);
  const body =
    jsonBytes(manifest) +
    completedKeys +
    reports +
    diagnostics +
    configurations +
    errors +
    stages +
    81_920;
  if (4 * Math.ceil(body / 3) + 1024 > MAX_ATTACHMENT_BYTES)
    throw new Error("suite exceeds encoded attachment budget before spend");
}

function parseModels(raw: string[] | string): string[] {
  const values = typeof raw === "string" ? raw.split(",") : raw;
  return [...new Set(values.map((value) => canonicalModelId(value.trim())))];
}

function fixedDesign() {
  return {
    bfclScenarios: 599,
    srScenarios: 600,
    srScenariosPerDataset: 100,
    srDatasets: 6,
    srDatasetNames: [...SR_DATASETS],
    seed: 42,
    repetitions: 1,
    llmPool: 100,
    llmTopK: 5,
    retrievalPools: { bfcl: [30, 100], sragents: [50, 100] },
    retrievalTopK: [1, 3, 5],
    arms: [...ARMS],
  };
}

function validateManifest(raw: unknown): asserts raw is SuiteManifest {
  const manifest = suiteManifestV1Schema.parse(raw);
  const allModels = [
    ...manifest.request.models,
    ...manifest.request.excludedModels,
    ...manifest.request.unmatchedExclusions,
  ];
  if (new Set(allModels).size !== allModels.length)
    throw new Error("manifest model selections and exclusions overlap");
  if (
    JSON.stringify(manifest.design) !== JSON.stringify(fixedDesign()) ||
    JSON.stringify(manifest.retrievers) !== JSON.stringify(RETRIEVERS)
  )
    throw new Error("manifest changes fixed suite design");
  validateScenarioIds(manifest.scenarioIds.bfcl, 599, !manifest.publishable, "BFCL");
  validateScenarioIds(manifest.scenarioIds.sragents, 600, !manifest.publishable, "SR-Agents");
  validateSrStratification(
    manifest.srDatasetByScenario,
    manifest.scenarioIds.sragents.length,
    !manifest.publishable,
  );
  if (jsonBytes(manifest.provenance) > 2048 || jsonBytes(manifest.release) > 1024)
    throw new Error("manifest provenance configuration exceeds cap");
  const keys = expectedWorkUnitKeysFor(
    manifest.request.models,
    manifest.scenarioIds.bfcl,
    manifest.scenarioIds.sragents,
  );
  if (
    keys.length !== manifest.expectedWorkUnitKeys.length ||
    keys.some((key, index) => key !== manifest.expectedWorkUnitKeys[index])
  )
    throw new Error("expected work-unit keys do not match frozen design");
}

/** Build the terminal stage recap. BM25 owns the once-per-model reused controls. */
export function suiteCoverageStages(
  manifest: SuiteManifest,
  completedKeys: readonly string[],
  incompleteStatus: Exclude<SuiteCoverageStage["status"], "completed"> = "skipped",
): SuiteCoverageStage[] {
  validateManifest(manifest);
  const completed = new Set(completedKeys);
  if (completed.size !== completedKeys.length)
    throw new Error("duplicate completed work-unit keys");
  const expectedKeys = new Set(manifest.expectedWorkUnitKeys);
  if ([...completed].some((key) => !expectedKeys.has(key)))
    throw new Error("completed work-unit key is outside the manifest");
  const stages = expectedStageKeys(manifest);
  return [...stages.entries()].map(([dimension, keys]) => {
    const [model, benchmark, retriever] = JSON.parse(dimension) as [
      string | null,
      "bfcl" | "sragents",
      Retriever,
    ];
    const count = keys.reduce((sum, key) => sum + Number(completed.has(key)), 0);
    return {
      model,
      benchmark,
      retriever,
      status: count === keys.length ? "completed" : incompleteStatus,
      completed: count,
      expected: keys.length,
    };
  });
}

function validateCoverageStages(
  manifest: SuiteManifest,
  coverage: SuiteResult["coverage"],
  completedKeys: ReadonlySet<string>,
): void {
  const expected = expectedStageKeys(manifest);
  const seen = new Set<string>();
  const remainder = { failed: 0, skipped: 0, cancelled: 0 };
  for (const stage of coverage.stages) {
    const dimension = stageDimension(stage.model, stage.benchmark, stage.retriever);
    const keys = expected.get(dimension);
    if (!keys || seen.has(dimension)) throw new Error("foreign or duplicate coverage stage");
    seen.add(dimension);
    const completed = keys.reduce((sum, key) => sum + Number(completedKeys.has(key)), 0);
    if (stage.expected !== keys.length || stage.completed !== completed)
      throw new Error("coverage stage counts do not match semantic work keys");
    if ((stage.status === "completed") !== (completed === keys.length))
      throw new Error("coverage stage status does not match completion");
    if (stage.status !== "completed") remainder[stage.status] += keys.length - completed;
  }
  if (seen.size !== expected.size) throw new Error("coverage omits requested stages");
  if (
    remainder.failed !== coverage.failed ||
    remainder.skipped !== coverage.skipped ||
    remainder.cancelled !== coverage.cancelled
  )
    throw new Error("terminal coverage is inconsistent with stage statuses");
}

function expectedStageKeys(manifest: SuiteManifest): Map<string, string[]> {
  const stages = new Map<string, string[]>();
  for (const key of manifest.expectedWorkUnitKeys) {
    const parts = key.split("|");
    const benchmark = parts[1] as "bfcl" | "sragents";
    const retriever =
      parts[0] === "R"
        ? (parts[4] as Retriever)
        : parts[6] === "-"
          ? "bm25"
          : (parts[6] as Retriever);
    const model = parts[0] === "R" ? null : manifest.request.models[Number(parts[3])];
    const dimension = stageDimension(model, benchmark, retriever);
    const keys = stages.get(dimension) ?? [];
    keys.push(key);
    stages.set(dimension, keys);
  }
  return stages;
}

function stageDimension(
  model: string | null,
  benchmark: "bfcl" | "sragents",
  retriever: Retriever,
): string {
  return JSON.stringify([model, benchmark, retriever]);
}

function stableVersion(version: string): boolean {
  return /^\d+\.\d+\.\d+$/.test(version);
}
function compareVersions(a: string, b: string): number {
  const aa = a.split(".").map(Number);
  const bb = b.split(".").map(Number);
  return aa[0] - bb[0] || aa[1] - bb[1] || aa[2] - bb[2];
}
function validateScenarioIds(ids: string[], expected: number, smoke: boolean, label: string): void {
  if (
    !Array.isArray(ids) ||
    ids.length === 0 ||
    ids.length > expected ||
    (!smoke && ids.length !== expected)
  )
    throw new Error(`${label} scenario count is not the fixed design`);
  if (new Set(ids).size !== ids.length || ids.some((id) => !/^[a-zA-Z0-9_.:-]{1,128}$/.test(id)))
    throw new Error(`${label} scenario IDs must be unique safe strings`);
}
function validateSrStratification(
  datasets: string[] | undefined,
  scenarios: number,
  smoke: boolean,
): void {
  if (smoke && (!datasets || datasets.length === 0)) return;
  if (!datasets || datasets.length !== scenarios)
    throw new Error("SR-Agents stratification needs a dataset for every scenario");
  const counts = new Map<string, number>();
  for (const dataset of datasets) {
    if (!(SR_DATASETS as readonly string[]).includes(dataset))
      throw new Error("SR-Agents stratification has an unknown dataset");
    counts.set(dataset, (counts.get(dataset) ?? 0) + 1);
  }
  if (!smoke && SR_DATASETS.some((dataset) => counts.get(dataset) !== 100))
    throw new Error("SR-Agents stratification requires 100 scenarios per dataset");
}
function expectedWorkUnitKeysFor(models: string[], bfcl: string[], sr: string[]): string[] {
  const keys: string[] = [];
  for (const [benchmark, scenarios, pools] of [
    ["bfcl", bfcl, [30, 100]],
    ["sragents", sr, [50, 100]],
  ] as const) {
    for (const [scenario] of scenarios.entries()) {
      for (const method of RETRIEVERS)
        for (const pool of pools)
          for (const k of [1, 3, 5])
            keys.push(`R|${benchmark}|${scenario}|0|${method}|${pool}|${k}`);
      for (const [model] of models.entries()) {
        for (const arm of ["control-baseline", "control-oracle"] as const)
          keys.push(
            `L|${benchmark}|${scenario}|${model}|0|${arm}|-|${arm === "control-oracle" ? "-" : 100}|5|0`,
          );
        for (const method of RETRIEVERS)
          keys.push(`L|${benchmark}|${scenario}|${model}|0|ratel-full|${method}|100|5|0`);
      }
    }
  }
  return keys;
}
function jsonBytes(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value), "utf8");
}
function checksum(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}
