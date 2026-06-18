# 8. Skill-retrieval eval mode for multi-tool queries

Date: 2026-06-18

## Status

Accepted. Extends ADR-0006's eval modes: mode (a) (MetaTool retrieval-only) is
split so that single-tool queries are scored as tool retrieval (as before) and
multi-tool queries are *additionally* scored as **skill retrieval**. The corpus,
pooling, and judge decisions of ADR-0006/0007 otherwise stand.

## Context

Ratel exposes not just tools but **skills** — a skill is a named bundle of
several tools for a task, surfaced from a separate `SkillRegistry` and returned
in its own bucket by the gateway (see ratel ADR-0012). ADR-0006 locked three
eval modes that measure *tool* retrieval only; there was no measurement of skill
retrieval. Since a multi-tool task is exactly what a skill is for, MetaTool's
multi-tool split is the natural place to evaluate it.

A multi-tool task is **all-or-nothing**: you need every one of its tools to
complete it. Two ways to retrieve that capability:

- **Tool retrieval** — surface the N individual tools. Succeeds only if *all N*
  land in the top-K (and they consume N of the K slots).
- **Skill retrieval** — surface the one skill bundle that carries all N tools.
  Succeeds if that single skill lands in the top-K (one slot, all N tools).

`ratel-ai-core` 0.2.0 publishes the real `SkillRegistry` (BM25 over
name + description + tags, same engine as the tool registry), so we can score
skill retrieval with the production engine rather than an approximation.

## Decision

### Ingest

Each MetaTool **multi-tool** query now yields **two** scenarios:

- **tool-retrieval** (`metatool-mt-<index>`, category `metatool-multi`) — the N
  gold tools, scored via `ToolRegistry`. Unchanged from ADR-0006.
- **skill-retrieval** (`metatool-skill-<index>`, category `metatool-skill`) — the
  gold tool set synthesized into one skill bundle, scored via `SkillRegistry`.
  Single gold = the bundle itself.

Single-tool queries are unchanged (one tool-retrieval scenario each).

### Synthetic skill construction

MetaTool ships no authored skills, so each bundle is **synthesized** from its
tools: `description` = the constituent tools' descriptions concatenated; `tags` =
the tool names (the skill indexer splits identifiers, so they become query
terms); `tools` = the tool ids (bundle membership, not indexed). The user query
is never copied in, so retrieval is not trivially self-matching. This is the
honest, deterministic, $0 option; it is also the main limitation (see Caveats).

### Aggregation — three buckets

The summary splits metrics into `single-tool · tool`, `multi-tool · tool`, and
`multi-tool · skill`, each carrying the same metric set. Skills draw distractors
from a skill-only universe (skills compete against skills, never tools).

### New metric — `complete_set_rate` / `complete_at_k`

Added so the two retrieval modes can be compared on a single, meaningful bar:
the fraction of queries where **every** gold item is in the top-K (per-row
`complete_at_k`; aggregated `complete_set_rate`). For single-gold buckets this
equals `hit_rate`; for multi-gold tool retrieval it is the strict "complete set
retrieved" rate.

### Provenance

The summary records the resolved `ratel_ai_core_version`, so the append-only
summary tracks how retrieval/skill metrics shift across engine updates.

## Caveats

- **Recall is not directly comparable across the two modes.** `multi-tool · tool`
  recall@K is **fractional** (the mean fraction of N gold tools in the top-K — it
  gives partial credit for a partial tool set). `multi-tool · skill` recall@K is
  **binary** (one gold bundle, 0 or 1). `hit@K` likewise differs in meaning. So a
  raw recall/hit gap between the two is **not** "the skill advantage." The
  directly comparable, task-meaningful metric is **`complete@K`** — "were all the
  required tools retrieved" — which is binary for both modes (all-N for tools, the
  bundle for skills). Any tool-vs-skill claim must be made on `complete@K`, and
  the rendered report states this inline.
- **Skills are synthetic.** Descriptions/tags are derived mechanically from the
  tools, not authored. The *direction* of results (skills win on complete-set) is
  robust; absolute numbers are indicative and would shift with a real authored
  skill catalog. This is the obvious follow-up if sharper skill numbers are needed.

## Consequences

- We can now report skill-retrieval quality alongside tool retrieval on the same
  multi-tool queries, on the production `SkillRegistry`.
- The headline multi-tool finding is framed on `complete@K`: surfacing one skill
  bundle delivers the complete toolset more often than fishing for all N tools.
- `ratel-ai-core` moves from `0.1.5` (tool-only) to `0.2.0` (adds `SkillRegistry`).
  The tool engine is unchanged across the bump (single-tool metrics are stable),
  so prior tool numbers remain comparable.
