// Builds the runtime pricing table from models.json, so per-model rates live in
// one place (the model definition) instead of a hardcoded table. Pricing is
// OPTIONAL: a model with no `pricing` — or one whose active backend has no rate —
// simply prices at $0, exactly as before. When present, the rate feeds the
// `dollar_cost` on every row and the `--dollar-global` cap.
//
// `pricing` on an entry is keyed by BACKEND, because the same model can be served
// from more than one stack at different prices:
//   { "bedrock": {…}, "anthropic": {…} }   (claude-*)
//   { "openai": {…} }                        (gpt-*)
// The active backend is derived from the model id + RATEL_LLM_BACKEND (the same
// env the AWS harness sets), so a `bedrock:` run reads `pricing.bedrock` and an
// `anthropic:` run reads `pricing.anthropic`. A flat ModelPrice (no backend key)
// is also accepted and applies to every backend.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ModelPrice, PricingTable } from "./metering.js";
import { REPO_ROOT } from "./paths.js";

type MaybeBackendPricing = Record<string, unknown>;

interface ModelEntry {
  id: string;
  pricing?: MaybeBackendPricing;
}

/** Location of the model catalog: MODELS_JSON override (used by the AWS harness),
 *  else models.json at the repo root (rides in with the checkout there too). */
function modelsJsonPath(): string {
  return process.env.MODELS_JSON || resolve(REPO_ROOT, "models.json");
}

/** Which backend serves this model id right now. Mirrors cli.ts's routing:
 *  claude-* → Bedrock when RATEL_LLM_BACKEND=bedrock, else the Anthropic API;
 *  gpt-* → OpenAI. Everything else (ollama, self-hosted links) has no vendor
 *  per-token price, so returns null. */
export function activeBackend(modelId: string): string | null {
  if (modelId.startsWith("claude")) {
    return process.env.RATEL_LLM_BACKEND === "bedrock" ? "bedrock" : "anthropic";
  }
  if (modelId.startsWith("gpt")) return "openai";
  return null;
}

/** True when `p` is a flat ModelPrice rather than a backend→ModelPrice map. */
function isModelPrice(p: unknown): p is ModelPrice {
  return typeof p === "object" && p !== null && typeof (p as ModelPrice).inputPer1M === "number";
}

/** Resolve one entry's `pricing` to a single ModelPrice for its active backend,
 *  or null when it has none. */
function priceForEntry(entry: ModelEntry): ModelPrice | null {
  const p = entry.pricing;
  if (!p) return null;
  if (isModelPrice(p)) return p; // flat: applies to any backend
  const backend = activeBackend(entry.id);
  if (backend && isModelPrice(p[backend])) return p[backend] as ModelPrice;
  return null;
}

/**
 * Build a PricingTable (model id → ModelPrice) from models.json for the current
 * run's backend. Missing/unreadable catalog or absent prices yield an empty
 * table — the run still works, every unpriced row just records $0. Never throws.
 */
export function loadModelPricing(path: string = modelsJsonPath()): PricingTable {
  let entries: ModelEntry[] = [];
  try {
    const catalog = JSON.parse(readFileSync(path, "utf8")) as { run?: ModelEntry[] };
    entries = catalog.run ?? [];
  } catch {
    return {}; // no catalog → no prices → $0 rows (unchanged behavior)
  }
  const table: PricingTable = {};
  for (const entry of entries) {
    if (!entry?.id) continue;
    const price = priceForEntry(entry);
    if (price) table[entry.id] = price;
  }
  return table;
}
