// Which `@ratel-ai/sdk` this process benchmarks, and how to load it.
//
// The benchmark measures a *specific* Ratel release, so the SDK is data, not a
// fixed import. Every value import of the SDK goes through `loadSdk()`; the rest
// of the codebase may still `import type` from it (types are erased, so they
// cost nothing at runtime and keep call sites readable).
//
// Phase 1 resolves the plain `@ratel-ai/sdk` specifier only — the same module
// the code imported directly before. Alias selection (`@ratel-ai/sdk-0.5.0`)
// arrives with `--sdk-version`; `select()` is the seam it will use.

import { createRequire } from "node:module";

const requirePkg = createRequire(import.meta.url);

/** Structural view of the SDK surface the benchmark consumes. Loaded shapes are
 *  checked with {@link supportsAsyncSearch} rather than by version string, so a
 *  prerelease (`0.5.0-rc.2`) routes on what it actually exposes. */
export type SdkModule = typeof import("@ratel-ai/sdk");

const DEFAULT_SPECIFIER = "@ratel-ai/sdk";

let specifier = DEFAULT_SPECIFIER;
let cached: Promise<SdkModule> | null = null;

/**
 * Point subsequent loads at a different installed copy of the SDK. Must be
 * called before the first {@link loadSdk}; throws afterwards, because a campaign
 * that measured two SDKs in one process would stamp rows it can't attribute.
 */
export function select(spec: string): void {
  if (cached && spec !== specifier) {
    throw new Error(
      `SDK already loaded as "${specifier}" — cannot switch to "${spec}" mid-process`,
    );
  }
  specifier = spec;
}

/** The module specifier currently selected. */
export function sdkSpecifier(): string {
  return specifier;
}

/**
 * Version of the selected SDK, read from its own `package.json`. Synchronous by
 * design: it is the `ratel_version` row dimension, and rows are built in hot
 * paths that shouldn't await a lookup this cheap.
 */
export function sdkVersion(): string {
  const pkg = requirePkg(`${specifier}/package.json`) as { version: string };
  return pkg.version;
}

/** Load (and memoize) the selected SDK. */
export function loadSdk(): Promise<SdkModule> {
  if (!cached) cached = import(specifier);
  return cached;
}

/**
 * Whether this SDK exposes the 0.5.0+ asynchronous retrieval API — `register()`
 * returning a promise and embedding inline, `searchAsync()` for semantic/hybrid,
 * and no `buildEmbeddings()`.
 *
 * Detected from the prototype rather than the version string: the two dialects
 * differ by what they expose, and prereleases carry the new API under an older-
 * sorting version. 0.4.0 and earlier have `buildEmbeddings` and no `searchAsync`.
 */
export function supportsAsyncSearch(mod: SdkModule): boolean {
  const proto = (mod.ToolCatalog as unknown as { prototype?: Record<string, unknown> })?.prototype;
  return typeof proto?.searchAsync === "function";
}

/** Reset module state. Tests only — production selects once at startup. */
export function resetForTests(): void {
  specifier = DEFAULT_SPECIFIER;
  cached = null;
}
