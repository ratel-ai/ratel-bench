import type { Scenario, ToolSpec } from "./types.js";

export function buildPool(
  allTools: ToolSpec[],
  expectedToolIds: string[],
  poolSize: number,
  seed = 0,
): ToolSpec[] {
  const expectedSet = new Set(expectedToolIds);
  const byId = new Map(allTools.map((t) => [t.id, t]));
  const expected = expectedToolIds.map((id) => byId.get(id)).filter((t): t is ToolSpec => !!t);
  const distractors = allTools.filter((t) => !expectedSet.has(t.id));
  shuffleInPlace(distractors, seed ^ poolSize);
  const slots = Math.max(0, poolSize - expected.length);
  return [...expected, ...distractors.slice(0, slots)];
}

/** Unique gold tools required across all of a scenario's turns. */
export function scenarioGoldIds(scenario: Scenario): string[] {
  const set = new Set<string>();
  for (const t of scenario.turns) set.add(t.expectedTool);
  return [...set];
}

/**
 * Pool for ONE scenario: its own gold tools first, then distractors. Distractors
 * vary per scenario (seed folds in the scenario id) so every scenario faces a
 * fresh, deterministic sample rather than one shared backdrop. This is what
 * makes pool-size sweeps meaningful on datasets where every tool is gold for
 * some scenario (e.g. MetaTool) — a global gold union would fill every pool.
 */
export function buildScenarioPool(
  allTools: ToolSpec[],
  scenario: Scenario,
  poolSize: number,
  seed = 0,
): ToolSpec[] {
  return buildPool(allTools, scenarioGoldIds(scenario), poolSize, seed ^ hashString(scenario.id));
}

/** FNV-1a 32-bit string hash, used to derive a per-scenario distractor seed. */
export function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function shuffleInPlace<T>(arr: T[], seed: number): void {
  const rng = mulberry32(seed >>> 0);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
