// ratel (full) — both Ratel layers active. BM25 top-K of the user prompt is
// pre-fetched from the catalog and registered as direct tools so the model
// sees its likely best moves up front; the `search_tools` / `invoke_tool`
// gateway tools are also exposed so the model can recover when pre-discovery
// missed the right tool. This is the canonical Ratel surface for ADR-0006
// mode (c) — every other Ratel arm is an ablation of this one.
//
// Read top-to-bottom this file is a worked example of "how to use
// `@ratel-ai/sdk` end-to-end": construct catalog → register pool → run
// `catalog.search` for pre-discovery → wire gateway tools → drive the AI
// SDK's tool loop.

import type { ToolCatalog } from "@ratel-ai/sdk";
import { toolCatalogMetadata } from "../../catalog-shapes.js";
import { embeddingArtifactOption } from "../../embedding-artifact.js";
import { buildToolCatalog, gatewayTools } from "../../sdk/adapter.js";
import type {
  AgentDescriptor,
  AgentRunInput,
  PrewarmInput,
  RetrievalMethod,
  Scenario,
  ToolSpec,
} from "../../types.js";
import {
  emptyToolBundle,
  registerDirect,
  registerGateway,
  runMeteredLoop,
  type ToolBundle,
} from "../_shared.js";

const ID = "ratel-full";

// Bundle cache so a cell's tool surface — and, for semantic/hybrid, its
// registration-time embedding — is built exactly ONCE and reused across every
// (model, run) that shares the same (scenario, pool, retriever). Populated by
// `prewarm` in the runner's serial pre-pass; `descriptor.run` reads from it.
// See `prewarm` for why this must not happen inside the concurrent metered loop.
const bundleCache = new Map<string, CachedBundle>();

/** A cached bundle plus the embedding-artifact path it was warmed from. */
interface CachedBundle {
  bundle: ToolBundle;
  embeddingArtifactPath: string | undefined;
}

/** Cache key: scenario + pool + retriever + seed. Seed is included because the
 *  pool is derived from it, so a different-seed run must never reuse a stale
 *  bundle. The embedding-artifact path is NOT in the key but is stored with the
 *  bundle: the direct tools are a search top-K over the warmed embeddings, so a
 *  later lookup with a different path (undefined vs path included) throws via
 *  `assertSameArtifact` rather than silently reusing the first ranking. */
function bundleKey(
  scenarioId: string,
  poolSize: number | null,
  retriever: RetrievalMethod,
  seed: number,
): string {
  return `${scenarioId}::${poolSize}::${retriever}::${seed}`;
}

/**
 * Pre-build (and, for semantic/hybrid, embed) each cell's bundle serially,
 * caching it. `ToolCatalog`'s embedding is a synchronous native (NAPI) call that
 * blocks the Node event loop; if it ran inside the concurrent agent loop, one
 * cell's embed would stall every other in-flight cell's `await`ed `generate()`,
 * inflating their `Date.now()`-based `wall_ms`. Doing it here, off the clock,
 * keeps `latency_p50_ms` honest — the timed loop then only pays the cheap
 * query-embed inside `search_tools`. Idempotent: re-prewarming a cached key with
 * the same embedding artifact path is a no-op; a different path throws.
 */
export async function prewarm(inputs: PrewarmInput[]): Promise<void> {
  for (const input of inputs) {
    const retriever = input.retriever ?? "bm25";
    const key = bundleKey(input.scenario.id, input.poolSize, retriever, input.seed);
    const cached = bundleCache.get(key);
    if (cached) {
      assertSameArtifact(key, cached, input.embeddingArtifactPath);
      continue;
    }
    const { bundle } = await buildRatelFullBundle({
      scenario: input.scenario,
      pool: input.pool,
      topK: input.topK,
      retriever,
      embeddingArtifactPath: input.embeddingArtifactPath,
    });
    bundleCache.set(key, { bundle, embeddingArtifactPath: input.embeddingArtifactPath });
  }
}

/** Clear the bundle cache. For tests that reuse the module across scenarios. */
export function resetPrewarmCache(): void {
  bundleCache.clear();
}

/** Whether a bundle is already cached for this cell — the lookup `descriptor.run`
 *  does before falling back to a cold build. Exposed for tests. */
export function isPrewarmed(cell: {
  scenario: { id: string };
  poolSize: number | null;
  retriever: RetrievalMethod;
  seed: number;
}): boolean {
  return bundleCache.has(
    bundleKey(cell.scenario.id, cell.poolSize, cell.retriever ?? "bm25", cell.seed),
  );
}

/**
 * Construct the catalog + AI SDK tool bundle for one ratel-full cell. Exposed
 * (rather than inlined into `descriptor.run`) so the integration logic — the
 * load-bearing piece of this arm — is unit-testable without spinning up an
 * agent loop. The `descriptor.run` below is then a thin wrapper.
 */
export async function buildRatelFullBundle(input: {
  scenario: Pick<Scenario, "prompt">;
  pool: ToolSpec[];
  topK: number;
  /** Retrieval method for both layers. Defaults to bm25 (model-free, no embeddings). */
  retriever?: RetrievalMethod;
  embeddingArtifactPath?: string;
}): Promise<{ bundle: ToolBundle; catalog: ToolCatalog }> {
  // 1. Catalog backs both layers: pre-discovery uses it for retrieval, and the
  //    gateway tools call back into the same instance to search/invoke at agent
  //    runtime — so the `method` set here governs pre-discovery AND the gateway.
  //
  //    Which SDK dialect builds it — sync `register` + `buildEmbeddings` below
  //    0.5.0, awaited `register` + `searchAsync` from 0.5.0 — is decided by
  //    `sdk/adapter.ts` from the installed SDK's own surface. The pre-0.5.0
  //    branch there is a verbatim copy of what this function used to do, so the
  //    0.2.0 / 0.3.0-rc.1 / 0.4.0 layers are unaffected.
  const method: RetrievalMethod = input.retriever ?? "bm25";
  const embeddingArtifact = await embeddingArtifactOption({
    path: input.embeddingArtifactPath,
    method,
  });
  const { catalog, search } = await buildToolCatalog({
    method,
    embeddingArtifact,
    tools: input.pool.map((spec) => ({
      ...toolCatalogMetadata(spec),
      execute: async () => ({ _stub: "stubbed for benchmark", toolId: spec.id }),
    })),
  });
  const { searchToolsTool, invokeToolTool } = await gatewayTools();

  const bundle = emptyToolBundle();

  // 2. Gateway: expose `search_tools` + `invoke_tool`. The agent can call
  //    these mid-loop to find/invoke any tool in the full pool — not just
  //    the pre-discovered top-K — which makes pre-discovery best-effort
  //    rather than load-bearing.
  registerGateway(searchToolsTool(catalog), bundle);
  registerGateway(invokeToolTool(catalog), bundle);

  // 3. Pre-discovery: BM25 top-K of the prompt, registered as direct tools.
  //    Cheap context-window win when retrieval gets it right; the gateway
  //    above is the safety net when it doesn't.
  for (const hit of await search(input.scenario.prompt, input.topK)) {
    const exec = catalog.getExecutable(hit.toolId);
    if (!exec) continue;
    const spec: ToolSpec = {
      id: exec.id,
      name: exec.name,
      description: exec.description,
      input_schema: exec.inputSchema as Record<string, unknown>,
      output_schema: (exec.outputSchema as Record<string, unknown>) ?? {},
    };
    registerDirect(spec, bundle);
  }

  return { bundle, catalog };
}

export const descriptor: AgentDescriptor = {
  id: ID,
  label: "ratel (full)",
  // Serial pre-pass: build + embed every cell's bundle before the concurrent
  // loop, so the blocking native embedding never inflates another cell's timer.
  prepare: (inputs) => prewarm(inputs),
  run: async (input: AgentRunInput) => {
    // Reuse the prewarmed bundle when present (the production path, so embedding
    // stayed off the clock); fall back to a cold build otherwise (unit tests /
    // any caller that skips prepare) — identical tool surface either way.
    const key = bundleKey(input.scenario.id, input.poolSize, input.retriever, input.seed);
    const cached = bundleCache.get(key);
    if (cached) assertSameArtifact(key, cached, input.embeddingArtifactPath);
    const bundle = cached?.bundle ?? (await buildRatelFullBundle(input)).bundle;
    return runMeteredLoop(ID, input, bundle);
  },
};

/** Throw when a cached cell is requested with a different embedding artifact
 *  than the one its bundle was warmed from. */
function assertSameArtifact(
  key: string,
  cached: CachedBundle,
  embeddingArtifactPath: string | undefined,
): void {
  if (cached.embeddingArtifactPath === embeddingArtifactPath) return;
  throw new Error(
    `ratel-full embedding artifact mismatch for cell ${key}: cached bundle was warmed ` +
      `from ${cached.embeddingArtifactPath ?? "<none>"}, got ${embeddingArtifactPath ?? "<none>"}`,
  );
}
