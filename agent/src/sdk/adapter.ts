// The only place the benchmark constructs a Ratel catalog.
//
// Two SDK dialects have to be spoken, and which one is in play is a property of
// the release under test — not of the caller:
//
//   < 0.5.0  sync `register()` per tool, explicit `buildEmbeddings()`, sync `search()`
//   >= 0.5.0 `await register(batch)` (embeds inline), `searchAsync()`, no `buildEmbeddings()`
//
// The legacy branch is a verbatim transcription of what the five call sites did
// before this module existed — same constructor choice, same per-tool register
// loop, same `buildEmbeddings()` placement, same sync `search()`. Published
// 0.2.0–0.4.0 layers are regression-gated against it, so treat that branch as
// frozen: changes there rewrite history.
//
// Both branches are exposed through one async signature. Catalogs are built in
// `prewarm()` (ratel-full.ts) and in the candidate generators — never inside the
// timed agent loop — so the extra microtask costs no measured latency.

import type { ExecutableTool, SearchHit, Skill, SkillHit, ToolCatalog } from "@ratel-ai/sdk";
import type { RetrievalMethod } from "../types.js";
import { loadSdk, type SdkModule, supportsAsyncSearch } from "./resolve.js";

/**
 * Embedding-model selection, kept structural on purpose: the concrete
 * `EmbeddingSpec` union only exists in 0.5.0's types, and this module has to
 * compile against whichever SDK is installed. Shape validation lives in
 * `embedding.ts`; by the time a value reaches here it has been checked.
 */
export type EmbeddingSpec = string | Record<string, unknown>;

/** A built catalog plus the retrieval call appropriate to its SDK dialect. */
export interface ToolCatalogHandle {
  /** The live catalog — gateway tools and `getExecutable` need the instance. */
  catalog: ToolCatalog;
  search(query: string, topK: number): Promise<SearchHit[]>;
}

/** SkillCatalog counterpart. Typed loosely because 0.4.0 and 0.5.0 disagree on
 *  `register`'s return, and the benchmark only ever calls `search`. */
export interface SkillCatalogHandle {
  catalog: unknown;
  search(query: string, topK: number): Promise<SkillHit[]>;
}

interface BuildOptions {
  /** Defaults to bm25 — the model-free method available on every SDK version. */
  method?: RetrievalMethod;
  /** 0.5.0+ only; rejected below that with the version that would be needed. */
  embedding?: EmbeddingSpec;
}

function assertEmbeddingSupported(mod: SdkModule, embedding: EmbeddingSpec | undefined): void {
  if (embedding === undefined || supportsAsyncSearch(mod)) return;
  throw new Error(
    "--embedding requires @ratel-ai/sdk >= 0.5.0 (the installed SDK has no configurable " +
      "embedding model). Select a 0.5.0+ SDK, or drop --embedding to use the built-in model.",
  );
}

/**
 * Build a tool catalog with `tools` registered and (for semantic/hybrid)
 * embeddings ready, then hand back a uniform async `search`.
 */
export async function buildToolCatalog(
  opts: BuildOptions & { tools: readonly ExecutableTool[] },
): Promise<ToolCatalogHandle> {
  const mod = await loadSdk();
  assertEmbeddingSupported(mod, opts.embedding);
  const method: RetrievalMethod = opts.method ?? "bm25";

  if (supportsAsyncSearch(mod)) {
    // 0.5.0+: options always passed (bm25 is its default anyway), batch register
    // so a semantic/hybrid catalog embeds the pool in one pass, and searchAsync
    // for every method — it is valid for bm25 too and keeps embedding work off
    // the event loop.
    const catalog = new mod.ToolCatalog({
      method,
      ...(opts.embedding === undefined ? {} : { embedding: opts.embedding }),
    } as ConstructorParameters<typeof mod.ToolCatalog>[0]);
    await (catalog.register as unknown as (t: readonly ExecutableTool[]) => Promise<void>).call(
      catalog,
      [...opts.tools],
    );
    return {
      catalog,
      search: (query, topK) =>
        (
          catalog as unknown as { searchAsync(q: string, k: number): Promise<SearchHit[]> }
        ).searchAsync(query, topK),
    };
  }

  // --- frozen: < 0.5.0. Do not modify without re-running the golden diff. ---
  const catalog = method === "bm25" ? new mod.ToolCatalog() : new mod.ToolCatalog({ method });
  for (const tool of opts.tools) catalog.register(tool);
  if (method !== "bm25") {
    (catalog as unknown as { buildEmbeddings(): void }).buildEmbeddings();
  }
  return { catalog, search: async (query, topK) => catalog.search(query, topK) };
}

/** {@link buildToolCatalog} for skills. Same dialect split, same frozen branch. */
export async function buildSkillCatalog(
  opts: BuildOptions & { skills: readonly Skill[] },
): Promise<SkillCatalogHandle> {
  const mod = await loadSdk();
  assertEmbeddingSupported(mod, opts.embedding);
  const method: RetrievalMethod = opts.method ?? "bm25";

  if (supportsAsyncSearch(mod)) {
    const catalog = new mod.SkillCatalog({
      method,
      ...(opts.embedding === undefined ? {} : { embedding: opts.embedding }),
    } as ConstructorParameters<typeof mod.SkillCatalog>[0]);
    await (catalog.register as unknown as (s: readonly Skill[]) => Promise<void>).call(catalog, [
      ...opts.skills,
    ]);
    return {
      catalog,
      search: (query, topK) =>
        (
          catalog as unknown as { searchAsync(q: string, k: number): Promise<SkillHit[]> }
        ).searchAsync(query, topK),
    };
  }

  // --- frozen: < 0.5.0 ---
  const catalog = method === "bm25" ? new mod.SkillCatalog() : new mod.SkillCatalog({ method });
  for (const skill of opts.skills) catalog.register(skill);
  if (method !== "bm25") {
    (catalog as unknown as { buildEmbeddings(): void }).buildEmbeddings();
  }
  return { catalog, search: async (query, topK) => catalog.search(query, topK) };
}

/** The SDK's gateway tools (`search_tools`, `invoke_tool`) and their ids.
 *  0.5.0 keeps both under `compat`, unchanged in behaviour, so the gateway
 *  surface the model sees stays identical across every layer. */
export async function gatewayTools(): Promise<
  Pick<SdkModule, "searchToolsTool" | "invokeToolTool">
> {
  const mod = await loadSdk();
  return { searchToolsTool: mod.searchToolsTool, invokeToolTool: mod.invokeToolTool };
}
