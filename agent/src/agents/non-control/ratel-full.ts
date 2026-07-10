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

import { invokeToolTool, searchToolsTool, ToolCatalog } from "@ratel-ai/sdk";
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
const bundleCache = new Map<string, ToolBundle>();

/** Cache key: a bundle is fully determined by scenario + pool + retriever. Seed
 *  is included because the pool is derived from it, so a different-seed run must
 *  never reuse a stale bundle. */
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
 * query-embed inside `search_tools`. Idempotent: re-prewarming a cached key is a
 * no-op, so it's safe to call once per campaign.
 */
export function prewarm(inputs: PrewarmInput[]): void {
  for (const input of inputs) {
    const retriever = input.retriever ?? "bm25";
    const key = bundleKey(input.scenario.id, input.poolSize, retriever, input.seed);
    if (bundleCache.has(key)) continue;
    const { bundle } = buildRatelFullBundle({
      scenario: input.scenario,
      pool: input.pool,
      topK: input.topK,
      retriever,
    });
    bundleCache.set(key, bundle);
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
export function buildRatelFullBundle(input: {
  scenario: Pick<Scenario, "prompt">;
  pool: ToolSpec[];
  topK: number;
  /** Retrieval method for both layers. Defaults to bm25 (model-free, no embeddings). */
  retriever?: RetrievalMethod;
}): { bundle: ToolBundle; catalog: ToolCatalog } {
  // 1. Catalog backs both layers: pre-discovery uses it for retrieval, and the
  //    gateway tools call back into the same instance to search/invoke at agent
  //    runtime — so the `method` set here governs pre-discovery AND the gateway.
  //
  //    `semantic`/`hybrid` are a 0.4.0+ capability (`ToolCatalog({ method })` +
  //    `buildEmbeddings()`). `bm25` is the default on every SDK version, so it
  //    uses the plain constructor — this keeps the 0.2.0 / 0.3.0-rc.1 code paths
  //    (older SDKs that lack the `method` option and `buildEmbeddings`) working
  //    exactly as before. Only touch the new API when a non-bm25 method is asked.
  const method: RetrievalMethod = input.retriever ?? "bm25";
  const catalog = method === "bm25" ? new ToolCatalog() : new ToolCatalog({ method });
  for (const spec of input.pool) {
    catalog.register({
      id: spec.id,
      name: spec.name,
      description: spec.description,
      inputSchema: spec.input_schema,
      outputSchema: spec.output_schema ?? {},
      execute: async () => ({ _stub: "stubbed for benchmark", toolId: spec.id }),
    });
  }
  // semantic/hybrid rank against an embedding cache built here; 0.4.0+ only.
  if (method !== "bm25") catalog.buildEmbeddings();

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
  for (const hit of catalog.search(input.scenario.prompt, input.topK)) {
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
    const bundle = bundleCache.get(key) ?? buildRatelFullBundle(input).bundle;
    return runMeteredLoop(ID, input, bundle);
  },
};
