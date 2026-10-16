import { mkdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import type { ExecutableTool, Skill } from "@ratel-ai/sdk";
import { loadSkillUniverse, toolCatalogMetadata } from "./catalog-shapes.js";
import { loadScenarios } from "./corpus.js";
import { resolveRepoPath } from "./paths.js";
import { buildToolUniverse } from "./pool.js";
import {
  acceptsEmbeddingArtifact,
  type EmbeddingSpec,
  supportsEmbeddingArtifact,
} from "./sdk/adapter.js";
import { loadSdk, sdkVersion } from "./sdk/resolve.js";
import type { RetrievalMethod } from "./types.js";

interface BuildEmbeddingArtifactOptions {
  benchmark: "bfcl" | "sragents";
  output: string;
  corpus?: string;
  skills?: string;
}

interface EmbeddingArtifactCapabilities {
  build: boolean;
  candidates: boolean;
  start: boolean;
}

interface EmbeddingArtifactOptionInput {
  /** `--embedding-artifact`; empty/undefined means no artifact. */
  path?: string;
  method: RetrievalMethod;
  /** `--embedding`; artifacts are always built with the SDK default model. */
  embedding?: EmbeddingSpec;
}

type ArtifactBuilder = (options: {
  output: string;
  tools?: readonly Omit<ExecutableTool, "execute">[];
  skills?: readonly Skill[];
}) => Promise<void>;

const artifactBytes = new Map<string, Promise<Uint8Array>>();

export class UnsupportedEmbeddingArtifactError extends Error {}

/** Build with the SDK selected through selectVersion, never the default import. */
export async function buildEmbeddingArtifact(
  options: BuildEmbeddingArtifactOptions,
): Promise<void> {
  const mod = await loadSdk();
  const build = (mod as unknown as { experimentalBuildEmbeddingArtifact?: ArtifactBuilder })
    .experimentalBuildEmbeddingArtifact;
  if (typeof build !== "function") {
    throw new UnsupportedEmbeddingArtifactError(
      `unsupported: SDK ${sdkVersion()} lacks experimentalBuildEmbeddingArtifact`,
    );
  }
  const input = resolveRepoPath(
    options.benchmark === "bfcl"
      ? (options.corpus ?? "test-data/bfcl-all.jsonl")
      : (options.skills ?? "test-data/sragents-skills.jsonl"),
  );
  const universe =
    options.benchmark === "bfcl"
      ? { tools: buildToolUniverse(loadScenarios(input)).map(toolCatalogMetadata) }
      : { skills: await loadSkillUniverse(input) };
  // An empty corpus still yields a valid header-only RAT1; never publish one.
  const [label, items] = universe.tools ? ["tools", universe.tools] : ["skills", universe.skills];
  if (!items?.length) {
    throw new Error(`empty ${options.benchmark} universe: no ${label} in ${input}`);
  }
  const output = resolveRepoPath(options.output);
  await mkdir(dirname(output), { recursive: true });
  await build({ output, ...universe });
}

/** End-to-end harness support for the resolver-selected SDK, without loading corpus data. */
export async function embeddingArtifactCapabilities(): Promise<EmbeddingArtifactCapabilities> {
  const mod = await loadSdk();
  const build =
    typeof (mod as unknown as { experimentalBuildEmbeddingArtifact?: unknown })
      .experimentalBuildEmbeddingArtifact === "function";
  // Same structural check the adapter applies before passing artifact bytes.
  const start = supportsEmbeddingArtifact(mod);
  const skillPrototype = mod.SkillCatalog.prototype as unknown as Record<string, unknown>;
  return {
    build,
    candidates: start && typeof skillPrototype.searchAsync === "function",
    start,
  };
}

/**
 * Catalog `embeddingArtifact` option for a CLI/prewarm caller. Checks SDK support
 * before reading bytes, so legacy SDKs warn once and embed inline even when the
 * path is missing; supporting SDKs still fail on a missing/invalid artifact.
 */
export async function embeddingArtifactOption(
  input: EmbeddingArtifactOptionInput,
): Promise<{ bytes: Uint8Array; onMiss: "embed" } | undefined> {
  if (!input.path) return undefined;
  if (input.embedding !== undefined) {
    throw new Error(
      "--embedding cannot be combined with --embedding-artifact: artifacts are built " +
        "with the SDK default model. Drop one of the two flags.",
    );
  }
  if (input.method === "bm25" || !acceptsEmbeddingArtifact(await loadSdk())) return undefined;
  return { bytes: await loadEmbeddingArtifact(input.path), onMiss: "embed" };
}

/** Load once per resolved path, sharing bytes across all catalogs in this process. */
export function loadEmbeddingArtifact(path: string): Promise<Uint8Array> {
  const resolved = resolve(resolveRepoPath(path));
  let bytes = artifactBytes.get(resolved);
  if (!bytes) {
    bytes = readFile(resolved);
    artifactBytes.set(resolved, bytes);
    // Don't memoize failures: a later call may find the file. The handler only
    // evicts; callers still observe the rejection through `bytes`.
    bytes.catch(() => artifactBytes.delete(resolved));
  }
  return bytes;
}
