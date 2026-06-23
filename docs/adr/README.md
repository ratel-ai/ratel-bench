# ADRs

Locked architectural decisions for the Ratel benchmark harness. Each ADR is immutable once `Accepted` — never edited, only superseded by a later ADR.

These three ADRs were written when the harness lived under `benchmark/` in the [`ratel-ai/ratel`](https://github.com/ratel-ai/ratel) monorepo. Path references inside (e.g. `benchmark/test-data/`, `benchmark/agent/`) reflect that original layout. After extraction into this standalone repo, the equivalent paths are simply rooted at the repo root: `benchmark/test-data/` → `test-data/`, `benchmark/agent/` → `agent/`, etc.

The ADRs are not edited to reflect the new layout because they document *decisions made at a moment in time*. The current repo layout is documented in [`../../README.md`](../../README.md) and the per-folder READMEs.

- [0005-benchmark-design.md](0005-benchmark-design.md) — overall harness (arms, models, variance, results storage)
- [0006-benchmark-corpus-and-eval-modes.md](0006-benchmark-corpus-and-eval-modes.md) — corpus pivot + the three eval modes
- [0007-benchmark-corpus-not-snapshotted.md](0007-benchmark-corpus-not-snapshotted.md) — corpus is ingested locally; no committed snapshot, no MetaTool sampling
- [0008-skill-retrieval-eval-mode.md](0008-skill-retrieval-eval-mode.md) — skill retrieval evaluated separately on an authored skill corpus (SR-Agents) via `SkillRegistry`
- [0009-bfcl-evaluation-and-export.md](0009-bfcl-evaluation-and-export.md) — BFCL function-calling eval + reproducible per-row / append-only summary / rebuildable multi-version report
