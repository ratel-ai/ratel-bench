// `--embedding` parsing and validation.
//
// Ratel 0.5.0 made the embedding model backing semantic/hybrid retrieval a
// runtime option, which is what lets one SDK build be benchmarked against
// several models. Before 0.5.0 the model was compiled into the native binary —
// which is why `0.3.0-semantic.1` (bge-small) and `0.3.0-semantic.2`
// (all-MiniLM) exist as separate layers in the report rather than as one layer
// run twice.
//
// The SDK accepts four mutually exclusive sources. We validate the shape here so
// a typo fails at argument-parse time with the accepted forms listed, rather
// than surfacing later as an opaque embedder error on the first scenario.

import type { EmbeddingSpec } from "./adapter.js";

/** Keys that select a source. Exactly one must be present. */
const SOURCE_KEYS = ["huggingface", "local", "ollama", "url"] as const;

/** Keys legal alongside a source (the SDK rejects cross-variant mixing itself,
 *  but listing them lets us catch outright typos before the SDK sees them). */
const MODIFIER_KEYS = [
  "revision",
  "download",
  "pooling",
  "queryPrefix",
  "docPrefix",
  "model",
  "apiKeyEnv",
] as const;

const USAGE = [
  "--embedding accepts either a local model directory path, or a JSON object:",
  "  /opt/models/bge                                  (local directory)",
  '  \'{"huggingface":"BAAI/bge-small-en-v1.5","download":true}\'',
  '  \'{"local":"/opt/models/bge"}\'',
  '  \'{"ollama":"nomic-embed-text"}\'',
  '  \'{"url":"https://…/v1/embeddings","model":"…","apiKeyEnv":"EMBED_KEY"}\'',
  "Optional on any variant: queryPrefix, docPrefix, pooling (cls|mean).",
].join("\n");

/**
 * Parse a raw `--embedding` argument.
 *
 * A value that doesn't start with `{` is taken as a local model directory path
 * and passed through as a bare string — the SDK's documented shorthand. Anything
 * else must be a JSON object naming exactly one source.
 */
export function parseEmbedding(raw: string | undefined): EmbeddingSpec | undefined {
  if (raw === undefined || raw === "") return undefined;

  const trimmed = raw.trim();
  if (!trimmed.startsWith("{")) return trimmed;

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (err) {
    throw new Error(`--embedding is not valid JSON (${(err as Error).message}).\n${USAGE}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`--embedding JSON must be an object.\n${USAGE}`);
  }

  const obj = parsed as Record<string, unknown>;
  const sources = SOURCE_KEYS.filter((k) => k in obj);
  if (sources.length === 0) {
    throw new Error(
      `--embedding names no source (expected one of: ${SOURCE_KEYS.join(", ")}).\n${USAGE}`,
    );
  }
  if (sources.length > 1) {
    throw new Error(
      `--embedding names ${sources.length} sources (${sources.join(", ")}) — exactly one is ` +
        `allowed.\n${USAGE}`,
    );
  }

  const known = new Set<string>([...SOURCE_KEYS, ...MODIFIER_KEYS]);
  const unknownKeys = Object.keys(obj).filter((k) => !known.has(k));
  if (unknownKeys.length) {
    throw new Error(`--embedding has unknown key(s): ${unknownKeys.join(", ")}.\n${USAGE}`);
  }

  // `url` is the only source needing a companion key; catching it here beats a
  // provider-side 4xx on the first embed.
  if ("url" in obj && typeof obj.model !== "string") {
    throw new Error(`--embedding {url:…} also requires "model".\n${USAGE}`);
  }

  return obj as EmbeddingSpec;
}

/** Human-readable label for a spec — used to name a run's version layer. */
export function describeEmbedding(spec: EmbeddingSpec | undefined): string {
  if (spec === undefined) return "default";
  if (typeof spec === "string") return spec;
  for (const k of SOURCE_KEYS) {
    const v = spec[k];
    if (typeof v === "string") return v;
  }
  return "custom";
}
