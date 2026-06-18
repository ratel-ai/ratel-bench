# 8. Skill-retrieval evaluation on an authored skill corpus (SR-Agents)

Date: 2026-06-18

## Status

Accepted. Adds a skill-retrieval experiment alongside the tool-retrieval modes
of ADR-0006/0007. Skill retrieval is evaluated on its own authored skill corpus
and is **fully separate** from tool retrieval — different data source, ingester,
run path, CLI subcommand, and output files.

## Context

Ratel exposes **skills** as well as tools. A skill is an authored
knowledge/procedure document, surfaced from a separate `SkillRegistry`: `name` +
`description` + `tags` drive BM25 ranking, `body` carries the markdown content
(the dispatch payload — **not** indexed), and `tools` is an *optional*,
customer-supplied "tools that pair well with this skill" edge (also not indexed).
A skill is **not** a bundle of tools.

An earlier version of this ADR synthesized "skills" by bundling the gold tools of
MetaTool multi-tool queries and scoring them against tool retrieval. That was
conceptually wrong — a synthesized tool-bundle is not an authored skill, and the
tool-vs-skill comparison measured nothing meaningful. MetaTool ships no authored
skills, so it cannot be a skills data source. That approach is removed entirely.

`ratel-ai-core` 0.2.0 publishes the real `SkillRegistry` (BM25 over
name + description + tags, same engine as the tool registry), so we score skill
retrieval with the production engine.

## Decision

### Data source — SR-Agents

Use [SR-Agents](https://github.com/oneal2000/SR-Agents) `data/bench/`:

- `corpus/corpus.json.zip` → ~26k authored skills, each
  `{ skill_id, name, description, content }`. Ingested into a **skill catalog**
  JSONL: `skill_id→id`, `name`, `description`, `content→body`; `tags`/`tools`
  empty. The catalog is the BM25 index and the distractor universe.
- `instances/<dataset>.json` → six datasets (`bigcodebench`, `champ`,
  `logicbench`, `medcalcbench`, `theoremqa`, `toolqa`), each instance
  `{ instance_id, dataset, question, skill_annotations:[ids], eval_data }`.
  Ingested into an **instances** JSONL: `prompt = question`,
  `gold_skill_ids = skill_annotations`. `eval_data.answer` is ignored
  (retrieval-only). Multi-mapping datasets (CHAMP) carry several gold ids — all
  count for Recall@K / nDCG@K. Instances whose gold ids are not all present in
  the catalog are skipped (counted).

`content→body` is carried in the catalog file for fidelity to production
registration, but `body` is **not** BM25-indexed, so it has no effect on
retrieval-only metrics; the run path loads the catalog without bodies to keep the
per-instance distractor universe cheap.

### Run path — separate from tools

A dedicated `skill-retrieval` CLI subcommand (`crate::skill_runner`) takes the
catalog + instances. For each instance the gold skills are resolved from the
catalog and pooled with distractors sampled from the catalog up to each
`pool_size` — the same gold-first, deterministically-shuffled pooling and the
same metric set as tool retrieval (recall/precision/hit/complete/MRR/nDCG + BM25
gold-score mean/median/stddev). Output and summary go to their own files
(`results/sragents-skill-*`). The tool `retrieval` path carries **no** skill
code: `Scenario` has no `candidate_skills`, and there is no skill bucket/mode.

### Aggregation — per dataset + aggregate

Results bucket per dataset (`subset = <dataset>`, `mode = "skill"`) plus an
aggregate `all` bucket, reusing the runner's accumulation/summary machinery. The
report renders one panel per dataset plus the aggregate.

### Provenance

The summary records the resolved `ratel_ai_core_version`, so the append-only
summary tracks how skill metrics shift across engine updates. `ratel-ai-core`
stays at `0.2.0` (the `SkillRegistry` is already present).

## Caveats

- **Recall is fractional for multi-mapping datasets.** When an instance has
  several gold skills (CHAMP), recall@K is the mean fraction retrieved;
  `complete@K` ("every gold skill in the top-K") is the all-or-nothing bar.
- **Only name + description are indexed** (catalog tags/tools are empty, body is
  never indexed), matching how the production `SkillRegistry` ranks.

## Consequences

- Skill-retrieval quality is reported on a real authored skill corpus, on the
  production `SkillRegistry`, independent of the tool-retrieval experiment.
- The invalid MetaTool synthetic-skill path (category `metatool-skill`,
  `build_skill_scenario`, the `multi-tool · skill` bucket) is removed; MetaTool
  is a tool corpus only. Prior tool numbers are unaffected.
- A new upstream dependency (SR-Agents) and a pure-Rust `zip` dependency (to
  unzip the corpus) are added; the upstream commit is pinned via the raw URLs in
  `ingest::sragents`.
