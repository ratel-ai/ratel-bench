// Builds the runtime pricing table from models.json, so per-model rates live in
// one place (the model definition) instead of a hardcoded table. Pricing is
// OPTIONAL: a model with no `pricing` — or one whose active backend has no rate —
// simply prices at $0, exactly as before. When present, the rate feeds the
// `dollar_cost` on every row and the `--dollar-global` cap.
//
// `pricing` on an entry is keyed by serving provider, because the same model can be served
// from more than one stack at different prices:
//   { "bedrock": {…}, "anthropic": {…} }   (claude-*)
//   { "openai": {…} }                        (gpt-*)
// The serving provider is derived from the qualified identity, so a Bedrock
// route reads `pricing.bedrock` and an Anthropic route reads `pricing.anthropic`.
// A flat ModelPrice (no provider key)
// is also accepted and applies to every backend.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ModelPrice, PricingTable } from "./metering.js";
import { canonicalModelId, parseModelIdentity } from "./model-identity.js";
import { REPO_ROOT } from "./paths.js";

type MaybeBackendPricing = Record<string, unknown>;

interface ModelEntry {
  id: string;
  aliases?: string[];
  pricing?: MaybeBackendPricing;
}

/** Build route-keyed prices from the shared model catalog. */
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
    for (const route of [entry.id, ...(entry.aliases ?? [])]) {
      const price = priceForEntry(entry, route);
      if (price) table[canonicalModelId(route)] = price;
    }
  }
  return table;
}

/** Location of the model catalog: MODELS_JSON override (used by the AWS harness),
 *  else models.json at the repo root (rides in with the checkout there too). */
export function modelsJsonPath(): string {
  return process.env.MODELS_JSON || resolve(REPO_ROOT, "models.json");
}

/** Serving provider from the model identity. Hosted and Ollama routes return null. */
export function activeBackend(modelId: string): string | null {
  const identity = parseModelIdentity(modelId);
  return identity.kind === "provider" ? identity.provider : null;
}

/** Resolve one entry's `pricing` to a single ModelPrice for its active backend,
 *  or null when it has none. */
function priceForEntry(entry: ModelEntry, route: string): ModelPrice | null {
  const p = entry.pricing;
  if (!p) return null;
  if (isModelPrice(p)) return p; // flat: applies to any backend
  const backend = activeBackend(route);
  if (backend && isModelPrice(p[backend])) return p[backend] as ModelPrice;
  return null;
}

/** True when `p` is a flat ModelPrice rather than a backend→ModelPrice map. */
function isModelPrice(p: unknown): p is ModelPrice {
  return typeof p === "object" && p !== null && typeof (p as ModelPrice).inputPer1M === "number";
}
