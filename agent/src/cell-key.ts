// Cell identities shared by the runner (resume + control cache) and the
// summarizers (supersede). A leaf module on purpose: importing it must not pull
// in the runner stack (agents, judges, the `@ratel-ai/sdk` native addon).

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
