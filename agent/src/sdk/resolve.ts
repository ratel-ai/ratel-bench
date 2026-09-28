// Which `@ratel-ai/sdk` this process benchmarks, and how to load it.
//
// The benchmark measures a *specific* Ratel release, so the SDK is data, not a
// fixed import. Every value import of the SDK goes through `loadSdk()`; the rest
// of the codebase may still `import type` from it (types are erased, so they
// cost nothing at runtime and keep call sites readable).
//
import { createRequire } from "node:module";

const requirePkg = createRequire(import.meta.url);

/** Structural view of the SDK surface the benchmark consumes. Loaded shapes are
 *  checked with {@link supportsAsyncSearch} rather than by version string, so a
 *  prerelease (`0.5.0-rc.2`) routes on what it actually exposes. */
export type SdkModule = typeof import("@ratel-ai/sdk");

const DEFAULT_SPECIFIER = "@ratel-ai/sdk";
export const GATEWAY_SEARCH_ID = "search_tools";
export const GATEWAY_INVOKE_ID = "invoke_tool";

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
 * Resolve a `--sdk-version` argument to a module specifier and select it.
 *
 * Versions are installed side by side as npm aliases (`@ratel-ai/sdk-0.5.0` →
 * `npm:@ratel-ai/sdk@0.5.0`), so one checkout can benchmark several releases
 * without reinstalling. `""` and `"local"` both mean the plain `@ratel-ai/sdk`
 * dependency — which is also what a `pnpm link`ed working copy of Ratel resolves
 * to, so an unreleased build is benchmarkable without adding an alias.
 *
 * Fails loudly on an unknown version: a silent fallback to the default SDK would
 * stamp rows with a version the run never actually exercised.
 */
export function selectVersion(version: string): void {
  if (!version || version === "local") {
    select(DEFAULT_SPECIFIER);
    return;
  }
  if (!/^\d+\.\d+\.\d+(?:-[\w.-]+)?$/.test(version)) {
    throw new Error(`--sdk-version needs an exact release (got "${version}")`);
  }
  const spec = `@ratel-ai/sdk-${version}`;
  try {
    requirePkg.resolve(`${spec}/package.json`);
  } catch {
    throw new Error(
      `--sdk-version ${version}: no alias "${spec}" installed. Add it to agent/package.json ` +
        `as "${spec}": "npm:@ratel-ai/sdk@${version}" and run pnpm install, or pass ` +
        `--sdk-version local to use the default @ratel-ai/sdk dependency.`,
    );
  }
  const declared = requirePkg("../../package.json") as {
    dependencies: Record<string, string>;
  };
  if (declared.dependencies[spec] !== `npm:@ratel-ai/sdk@${version}`) {
    throw new Error(
      `--sdk-version ${version}: alias "${spec}" must pin npm:@ratel-ai/sdk@${version}`,
    );
  }
  select(spec);
}

/**
 * Version of the selected SDK, read from its own `package.json`. Synchronous by
 * design: it is the `ratel_version` row dimension, and rows are built in hot
 * paths that shouldn't await a lookup this cheap.
 */
export function sdkVersion(): string {
  const pkg = requirePkg(`${specifier}/package.json`) as { name: string; version: string };
  const requested = specifier.match(/^@ratel-ai\/sdk-(.+)$/)?.[1];
  if (pkg.name !== "@ratel-ai/sdk" || (requested && pkg.version !== requested)) {
    throw new Error(
      `SDK identity mismatch: selected ${specifier}, loaded ${pkg.name}@${pkg.version}`,
    );
  }
  return pkg.version;
}

/** Load (and memoize) the selected SDK. */
export function loadSdk(): Promise<SdkModule> {
  if (!cached) {
    sdkVersion();
    cached = import(specifier).then((mod) => {
      assertSdkApi(mod);
      return mod;
    });
  }
  return cached;
}

/** Fail at campaign startup when a selected release cannot run benchmark catalogs. */
export async function validateSdk(): Promise<void> {
  await loadSdk();
}

export function assertSdkApi(mod: SdkModule): void {
  const api = mod as unknown as Record<string, unknown>;
  for (const name of ["ToolCatalog", "SkillCatalog", "searchToolsTool", "invokeToolTool"]) {
    if (typeof api[name] !== "function") {
      throw new Error(`unsupported SDK API in ${specifier}: missing ${name}`);
    }
  }
  if (api.SEARCH_TOOLS_ID !== GATEWAY_SEARCH_ID || api.INVOKE_TOOL_ID !== GATEWAY_INVOKE_ID) {
    throw new Error(`unsupported SDK API in ${specifier}: search_tools/invoke_tool ids changed`);
  }
  for (const name of ["ToolCatalog", "SkillCatalog"] as const) {
    const proto = (api[name] as { prototype: Record<string, unknown> }).prototype;
    if (
      typeof proto?.register !== "function" ||
      (typeof proto?.searchAsync !== "function" && typeof proto?.search !== "function")
    ) {
      throw new Error(`unsupported SDK API in ${specifier}: ${name} needs register and search`);
    }
  }
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
