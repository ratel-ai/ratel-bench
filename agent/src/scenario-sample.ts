import type { Scenario } from "./types.js";

/**
 * Seeded shuffle so a `--scenarios N` subset is representative of the full
 * corpus rather than an id-sorted prefix (which on MetaTool would cluster all
 * `metatool-mt-*` rows at the head). Same `seed` reproduces the same subset
 * across runs — important for resume.
 */
export function sampleScenarios(
  scenarios: Scenario[],
  limit: number | undefined,
  seed: number,
): Scenario[] {
  if (limit === undefined || limit >= scenarios.length) return scenarios;
  const shuffled = [...scenarios];
  let h = seed >>> 0;
  // Fisher-Yates with a mulberry32 PRNG seeded from `seed`.
  const rng = () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, limit);
}
