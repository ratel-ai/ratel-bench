import { mkdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import type { ExecutableTool, Skill } from "@ratel-ai/sdk";
import { loadSkillUniverse, toolCatalogMetadata } from "./catalog-shapes.js";
import { loadScenarios } from "./corpus.js";
import { resolveRepoPath } from "./paths.js";
import { buildToolUniverse } from "./pool.js";
import { loadSdk, sdkVersion } from "./sdk/resolve.js";

interface BuildEmbeddingArtifactOptions {
  benchmark: "bfcl" | "sragents";
  output: string;
  corpus?: string;
  skills?: string;
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
  const universe =
    options.benchmark === "bfcl"
      ? {
          tools: buildToolUniverse(
            loadScenarios(resolveRepoPath(options.corpus ?? "test-data/bfcl-all.jsonl")),
          ).map(toolCatalogMetadata),
        }
      : {
          skills: await loadSkillUniverse(
            resolveRepoPath(options.skills ?? "test-data/sragents-skills.jsonl"),
          ),
        };
  const output = resolveRepoPath(options.output);
  await mkdir(dirname(output), { recursive: true });
  await build({ output, ...universe });
}

/** Load once per resolved path, sharing bytes across all catalogs in this process. */
export function loadEmbeddingArtifact(path: string): Promise<Uint8Array> {
  const resolved = resolve(resolveRepoPath(path));
  let bytes = artifactBytes.get(resolved);
  if (!bytes) {
    bytes = readFile(resolved);
    artifactBytes.set(resolved, bytes);
  }
  return bytes;
}
