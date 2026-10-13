// Cell identities shared by the runner (resume + control cache) and the
// summarizers (supersede). A leaf module on purpose: importing it must not pull
// in the runner stack (agents, judges, the `@ratel-ai/sdk` native addon).

import { canonicalModelId, SERVING_PROVIDERS } from "./model-identity.js";
import type { Arm, CellResult } from "./types.js";

export interface CellKey {
  ratelVersion: string;
  scenarioId: string;
  arm: Arm;
  model: string;
  runIndex: number;
  /** `null` for pool-size-agnostic arms — drops the `::p<n>` suffix from the key. */
  poolSize: number | null;
}

export function cellKeyString(k: CellKey): string {
  const base = `${k.ratelVersion}::${k.scenarioId}::${k.arm}::${k.model}::${k.runIndex}`;
  return k.poolSize === null ? base : `${base}::p${k.poolSize}`;
}

/** A row's resume identity: version + scenario + arm + model + run (+ pool). */
export function cellKeyOf(cell: CellResult): string {
  return cellKeyString({
    ratelVersion: cell.ratel_version,
    scenarioId: cell.scenario_id,
    arm: cell.arm,
    model: cell.model,
    runIndex: cell.run_index,
    poolSize: cell.pool_size,
  });
}

/** Version-agnostic identity for a control cell — reused across ratel versions. */
export function controlKeyString(k: Omit<CellKey, "ratelVersion">): string {
  const base = `${k.scenarioId}::${k.arm}::${k.model}::${k.runIndex}`;
  return k.poolSize === null ? base : `${base}::p${k.poolSize}`;
}

export function controlKeyOf(cell: CellResult): string {
  return controlKeyString({
    scenarioId: cell.scenario_id,
    arm: cell.arm,
    model: cell.model,
    runIndex: cell.run_index,
    poolSize: cell.pool_size,
  });
}

/** Bare historical names need recorded route evidence; unknown stays isolated. */
export function modelRouteOfRow(row: {
  model: string;
  provider?: string;
  serving_provider?: string;
}): string {
  if (/^(bedrock|anthropic|openai|gcp|xai)[/:]/.test(row.model)) {
    return canonicalModelId(row.model);
  }
  if (/^(https?:\/\/|ollama:)/.test(row.model)) return row.model;
  const evidence = historicalProvider(row.provider);
  if (
    row.serving_provider &&
    (!(SERVING_PROVIDERS as readonly string[]).includes(row.serving_provider) ||
      (evidence && evidence !== row.serving_provider))
  )
    return row.model;
  const provider = row.serving_provider ?? evidence;
  return provider ? `${provider}/${row.model}` : row.model;
}

function historicalProvider(provider: string | undefined): string | undefined {
  if (provider === "amazon-bedrock" || provider?.startsWith("bedrock-mantle.")) return "bedrock";
  if (provider === "anthropic.messages") return "anthropic";
  if (provider?.startsWith("vertex.") || provider?.startsWith("google.vertex.")) return "gcp";
  if (provider === "xai.responses") return "xai";
  // openai.responses is also used by Bedrock's OpenAI-compatible API.
  return undefined;
}
