// Per-call output caps (`maxOutputTokens`). One source of truth: models.json
// `run[].maxOutputTokens`, overridable per run with `--max-output-tokens N|none`.
// No silent default: an id with no catalog cap resolves to `null` — no cap is
// sent and the provider default applies — and the CLIs log it (`capsLine`).
//
// The cap is requested per call (ToolLoopAgent / generateObject settings), not
// via `wrapLanguageModel` middleware, and stamped on every row as
// `max_output_tokens`. Cache reuse and resume key on it (`cacheTier`,
// `checkResumeCaps`), so rows produced under different recorded caps never mix
// silently; rows with no recorded cap (legacy, pre-cap builds) are the one
// tolerated exception. An explicit `--max-output-tokens` neither serves them from
// the cache nor resumes over ones an earlier invocation served into the output;
// live legacy rows only warn, with or without the flag.

import { existsSync, readFileSync } from "node:fs";
import { isInfraError } from "./cell-errors.js";
import { providerOf } from "./metering.js";
import { canonicalModelId } from "./model-identity.js";
import { modelsJsonPath } from "./pricing.js";
import type { AttemptRow } from "./rerun.js";
import type { ResolvedModel, RunnerModel } from "./types.js";

/** The models.json fields caps read (the file carries more, e.g. pricing). */
export interface ModelCatalogEntry {
  id: string;
  /** Alternate route IDs retained for historical callers. */
  aliases?: string[];
  /** `<baseURL>#<model-name>` for self-hosted models; matched against the run's id. */
  endpoint?: string;
  bedrockProfile?: string;
  /** Positive integer; absent = no cap sent. */
  maxOutputTokens?: number;
}

/** `--max-output-tokens`: a positive integer, or `"none"` to send no cap. */
export type OutputCapOverride = number | "none";

/** How `checkResumeCaps` names a reused row with no recorded cap. */
const LEGACY_LABEL = "legacy (no recorded cap)";

/** The cache-relevant harness a row was produced under. */
export interface Harness {
  provider?: string;
  max_output_tokens?: number | null;
}

/** A row as `checkResumeCaps` sees it (BFCL `CellResult` and SR cells both fit). */
export interface ResumeRow extends AttemptRow, Harness {
  model: string;
}

/**
 * The cap to request for `modelId`: the override when given (`"none"` → null),
 * else the catalog entry's `maxOutputTokens` (entry matched by `id`, or by
 * `endpoint` for `<url>#name` ids), else `null`. Throws on an invalid override
 * or catalog value rather than running with a surprise cap.
 */
export function resolveOutputCap(
  modelId: string,
  catalog: readonly ModelCatalogEntry[],
  override?: OutputCapOverride,
): number | null {
  if (override === "none") return null;
  if (override !== undefined) {
    return positiveInt(override, `--max-output-tokens must be a positive integer or "none"`);
  }
  const entry = findModelCatalogEntry(modelId, catalog);
  if (entry?.maxOutputTokens === undefined) return null;
  return positiveInt(
    entry.maxOutputTokens,
    `models.json: ${entry.id}.maxOutputTokens must be a positive integer`,
  );
}

/** Look up a catalog entry by serving identity, alias, or hosted endpoint. */
export function findModelCatalogEntry(
  modelId: string,
  catalog: readonly ModelCatalogEntry[],
): ModelCatalogEntry | undefined {
  const canonical = canonicalModelId(modelId);
  return catalog.find(
    (entry) =>
      canonicalModelId(entry.id) === canonical ||
      entry.aliases?.some((alias) => canonicalModelId(alias) === canonical) ||
      entry.endpoint === canonical,
  );
}

/**
 * models.json `run[]` entries. A missing catalog is empty (every id → no cap);
 * an unparseable one throws, since it would otherwise drop every cap silently.
 */
export function loadModelCatalog(path: string = modelsJsonPath()): ModelCatalogEntry[] {
  if (!existsSync(path)) return [];
  let catalog: { run?: ModelCatalogEntry[] };
  try {
    catalog = JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    throw new Error(`models.json at ${path} is not valid JSON: ${(err as Error).message}`);
  }
  if (!Array.isArray(catalog.run)) {
    if (catalog.run === undefined) return [];
    throw new Error(`models.json at ${path}: run must be an array`);
  }
  const owners = new Map<string, number>();
  for (const [index, entry] of catalog.run.entries()) {
    if (!entry || typeof entry.id !== "string") {
      throw new Error(`models.json at ${path}: every run entry needs a model id`);
    }
    if (entry.aliases !== undefined && !Array.isArray(entry.aliases)) {
      throw new Error(`models.json at ${path}: ${entry.id}.aliases must be an array`);
    }
    const routes = [
      entry.id,
      ...(entry.aliases ?? []),
      ...(entry.endpoint ? [entry.endpoint] : []),
    ];
    for (const route of routes) {
      const canonical = canonicalModelId(route);
      const owner = owners.get(canonical);
      if (owner !== undefined && owner !== index) {
        throw new Error(`models.json at ${path}: duplicate catalog route ${canonical}`);
      }
      owners.set(canonical, index);
    }
  }
  return catalog.run;
}

/**
 * Resolve each id, THEN attach its cap. Attaching after resolution (instead of
 * inside the resolver) means a resolver branch that returns early — e.g. the
 * AWS harness's injected Bedrock guard, which returns a bare `{id, model}` —
 * cannot skip the cap.
 */
export function buildRunnerModels(
  modelIds: readonly string[],
  resolve: (modelId: string) => ResolvedModel,
  // `override` is a required key (possibly undefined) so a CLI can't drop its flag silently.
  opts: { catalog: readonly ModelCatalogEntry[]; override: OutputCapOverride | undefined },
): RunnerModel[] {
  return modelIds.map((modelId) => {
    const canonicalId = canonicalModelId(modelId);
    const { model } = resolve(canonicalId);
    return {
      id: canonicalId,
      model,
      maxOutputTokens: resolveOutputCap(canonicalId, opts.catalog, opts.override),
    };
  });
}

/**
 * Startup log line, e.g. `caps: bedrock/claude-haiku-4-5=4096, ollama:qwen3.5=none (no
 * models.json cap)`. Uncapped models are called out, never implied.
 */
export function capsLine(models: readonly RunnerModel[], override?: OutputCapOverride): string {
  const why = override === "none" ? "--max-output-tokens none" : "no models.json cap";
  const parts = models.map((m) =>
    m.maxOutputTokens === null ? `${m.id}=none (${why})` : `${m.id}=${m.maxOutputTokens}`,
  );
  return `caps: ${parts.join(", ")}`;
}

/** The harness a model runs under now: its SDK provider id and requested cap. */
export function harnessOf(m: Pick<RunnerModel, "model" | "maxOutputTokens">): Harness {
  return { provider: providerOf(m.model), max_output_tokens: m.maxOutputTokens };
}

/** `harnessOf` for every model of a run, keyed by model id. */
export function harnessByModel(models: readonly RunnerModel[]): Map<string, Harness> {
  return new Map(models.map((m) => [m.id, harnessOf(m)]));
}

/**
 * How a cached control row may serve the current harness:
 *  - `exact`:  same provider and same requested cap (`null` = uncapped matches
 *    `null`). A same-provider row with no recorded cap also counts for an
 *    uncapped run: pre-cap builds sent no cap either, so the request was identical;
 *  - `legacy`: a row with no recorded cap (pre-cap builds) whose provider is
 *    unknown or matches. Its effective cap is unknown (the SDK's implicit default);
 *  - `null`:   never — a different recorded provider or cap would mix
 *    backends/caps silently.
 * `allowLegacy: false` — any explicit `--max-output-tokens` (N or `none`) —
 * drops the legacy tier: such runs must measure the cap they name.
 */
export function cacheTier(
  row: Harness,
  current: Harness | undefined,
  allowLegacy = true,
): "exact" | "legacy" | null {
  if (row.max_output_tokens === undefined) {
    if (row.provider !== undefined && current && row.provider !== current.provider) return null;
    if (row.provider !== undefined && current?.max_output_tokens === null) return "exact";
    return allowLegacy ? "legacy" : null;
  }
  if (
    current &&
    row.provider === current.provider &&
    row.max_output_tokens === current.max_output_tokens
  ) {
    return "exact";
  }
  return null;
}

/**
 * The better of two cache candidates for one key under `current`: an exact-tier
 * row beats a legacy one; within a tier the earliest `generated_at` wins (the
 * original baseline). Incompatible candidates never win. So a REBASELINE
 * (`--force`) run's exact rows replace older legacy rows in what is served.
 */
export function preferCacheRow<T extends Harness & { generated_at?: string }>(
  prev: T | undefined,
  cand: T,
  current: Harness | undefined,
  allowLegacy = true,
): T | undefined {
  const candTier = cacheTier(cand, current, allowLegacy);
  if (candTier === null) return prev;
  if (!prev) return cand;
  const prevTier = cacheTier(prev, current, allowLegacy);
  if (candTier !== prevTier) return candTier === "exact" ? cand : prev;
  return (cand.generated_at ?? "") < (prev.generated_at ?? "") ? cand : prev;
}

/**
 * Guard a resume against mixing caps. `rows` are the output's rows at the
 * current version; `harness` maps each model of this run to its harness (the cap
 * compared is its `max_output_tokens`). Throws when a row of a model in this run
 * recorded a different cap (`null` included) — live rows, and reused rows too: a
 * reused row with a recorded cap was served exact under the cap of an earlier
 * invocation, so a mismatch means this label already ran under another cap.
 * Reused rows with no recorded cap (legacy-tier serves) are skipped — unless
 * `allowLegacy` is false (an explicit `--max-output-tokens`, which serves exact
 * rows only) and the row isn't exact for this harness (`cacheTier`): keeping it
 * would leave an unknown-cap control under a label that names its cap. Also
 * skipped: infra-error rows (transient/access: unmeasured, excluded from metrics
 * and cap provenance). Returns how many LIVE rows predate the field (legacy):
 * those resume with a warning, not an error.
 */
export function checkResumeCaps(
  rows: readonly ResumeRow[],
  harness: ReadonlyMap<string, Harness>,
  allowLegacy = true,
): { legacy: number } {
  let legacy = 0;
  const conflicts = new Map<string, Set<string>>();
  const conflict = (model: string, found: string) => {
    (conflicts.get(model) ?? conflicts.set(model, new Set()).get(model))?.add(found);
  };
  for (const r of rows) {
    const current = harness.get(r.model);
    if (!current || isInfraError(r)) continue;
    if (r.max_output_tokens === undefined) {
      if (r.cache_source !== "reused") legacy++;
      else if (!allowLegacy && cacheTier(r, current, false) === null) {
        conflict(r.model, LEGACY_LABEL);
      }
      continue;
    }
    if (r.max_output_tokens !== current.max_output_tokens) {
      conflict(r.model, capLabel(r.max_output_tokens));
    }
  }
  if (conflicts.size > 0) {
    const detail = [...conflicts]
      .map(
        ([model, found]) =>
          `${model}: output has ${[...found].join("/")}, ` +
          `run uses ${capLabel(harness.get(model)?.max_output_tokens)}`,
      )
      .join("; ");
    const legacyNote = [...conflicts.values()].some((f) => f.has(LEGACY_LABEL))
      ? " An explicit --max-output-tokens serves exact-tier rows only, so it can't keep " +
        "legacy controls an earlier invocation served into this output (drop the flag to keep them)."
      : "";
    throw new Error(
      `resume: the output holds rows (live or reused) at this version produced under a ` +
        `different max_output_tokens (${detail}). Resuming would mix caps within one label — ` +
        `use the original cap, a fresh --output, or --force.${legacyNote}`,
    );
  }
  return { legacy };
}

/**
 * `checkResumeCaps` for a run's `models`, logging the one legacy warning. Callers
 * pass only the output's rows at `version` (their version fields differ), and the
 * run's `allowLegacy` (false under an explicit `--max-output-tokens`).
 */
export function guardResumeCaps(
  rows: readonly ResumeRow[],
  models: readonly RunnerModel[],
  version: string,
  allowLegacy: boolean,
): void {
  const { legacy } = checkResumeCaps(rows, harnessByModel(models), allowLegacy);
  if (legacy > 0) {
    console.warn(
      `warn: resume: ${legacy} live row(s) at ${version} predate max_output_tokens ` +
        `(cap unknown); resuming over them with this run's caps`,
    );
  }
}

/** A cap for logs and errors: `null` (uncapped) reads `none`, like the flag. */
function capLabel(cap: number | null | undefined): string {
  return cap == null ? "none" : String(cap);
}

function positiveInt(value: unknown, message: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    const got = typeof value === "string" ? JSON.stringify(value) : String(value);
    throw new Error(`${message} (got ${got})`);
  }
  return value;
}
