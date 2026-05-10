# BFCL-Loaded — Methodology

Experimental design for measuring how a context-engineering layer (Ratel) recovers performance lost to tool-pool noise, built on top of the Berkeley Function-Calling Leaderboard.

## Construct

What we measure: **the cost of tool-pool noise on agent performance, and how much of that cost a context-engineering layer recovers**, holding the model and prompt constant.

What we do not measure as the construct: which model is best at function calling; how good Ratel is at compression in the abstract. Tool-selection quality is reported as a secondary diagnostic (see Metrics) but does not stand in for agent outcome — a high selection score with a flat agent-outcome delta still means Ratel did not help.

## Three-condition design

For each scenario, the same model runs three conditions on the same items:

| Condition  | Tools delivered to model                       | Interpretation                  |
|------------|-------------------------------------------------|---------------------------------|
| Oracle     | Only the necessary tools (BFCL default)         | Upper bound — perfect retriever |
| Baseline   | Loaded pool (necessary + distractors)           | What the user faces today       |
| Treatment  | Loaded pool, filtered by Ratel                  | What Ratel delivers             |

The headline numbers are deltas, not absolute scores:

- **Oracle − Baseline** — cost of noise (the problem)
- **Treatment − Baseline** — Ratel's lift
- **Oracle − Treatment** — headroom remaining

Significance: paired bootstrap on items, k ≥ 5 runs per item per condition. Memorization of BFCL by the model under test affects all three conditions equally on shared items, so the deltas are robust to contamination; the absolute oracle score is not. Writeups must state this.

## Dataset

- Source: BFCL v3.
- Headline category: **multi-turn**. Single-turn static categories (simple, multiple, parallel) saturate quickly under modern frontier models and discriminate compression methods poorly.
- Reported: every category, broken out. No averaging across categories — they measure different things.

## Tool-pool construction

A loaded pool for a scenario is the scenario's necessary tools ∪ a sampled distractor set drawn from a candidate corpus. The construction is the experiment.

### Controlled property: intra-pool pairwise similarity distribution

For a pool of N tools, we have N(N−1)/2 pairwise tool–tool cosine similarities. The shape of this distribution is the property we control across pool sizes.

We control intra-pool pairwise similarity (rather than distractor-to-necessary similarity) because:

- It is not biased toward the benchmark's specific scenarios — it characterises the pool itself, independent of any ground-truth answer.
- It corresponds to a measurable property of any real-world tool catalog, with no labelling required, so we can compare synthetic pools to real MCP catalogs directly.
- Distractor-to-necessary difficulty falls out as a consequence: a pool with high intra-pool similarity that includes a necessary tool will, on average, contain distractors close to that necessary tool.

### Target distribution

Default: truncated normal on [0, 1] with documented (μ, σ). Parametric form chosen so μ and σ act as explicit difficulty knobs for sweeps. Expected to be revisited (see Open questions).

For every reported pool size N ∈ {30, 50, 100, 200, 500}, the empirical pairwise similarity distribution of each constructed pool must match the target within a tolerance (Kolmogorov–Smirnov D below a threshold; threshold itself set empirically). This gives us a curve along N where size is the only thing varying.

### Candidate corpus

To match arbitrary target distributions at N = 500, the pool we draw from must be substantially larger. Minimum 2,000 tools, sourced from:

- BFCL's own tool set (all categories)
- LLM-generated synthetic tools, prompted to fill thin regions of the target similarity distribution

The corpus is built once, versioned, and committed alongside this methodology. Real MCP-server tools are deferred to a follow-up calibration pass — see Open questions.

### Controller embedder

Pairwise similarity is computed as cosine similarity over an external reference embedder, fixed for the run (default TBD: candidate options include `sentence-transformers/all-mpnet-base-v2` and OpenAI `text-embedding-3-small`).

**The controller embedder is walled off from Ratel.** Ratel must not use it at any layer. Otherwise we use the same ruler to control difficulty and to do tool selection, which collapses the experiment.

### Per-(scenario, N, target) sampling

For each (scenario, N, target) triple, sample m ≥ 3 distinct pools, each independently satisfying the distribution-match tolerance. Variance across pools is reported as a separate axis from variance across runs — pool composition is its own source of uncertainty.

### Multi-necessary and multi-turn defaults

- BFCL multi-turn scenarios have different necessary tool sets per turn. The "necessary set" used for pool construction is the **union across all turns** in the scenario.
- Where a derived metric needs a single similarity-to-necessary scalar (never pool construction itself), aggregate across multiple necessary tools using **max** — what a retriever would surface.

Both are defaults, exposed as parameters.

## Baselines

In addition to oracle (upper bound) and Ratel (treatment):

- **No-op** — full pool passed to the model, no filtering. This is the Baseline condition above; restated here as the lower bar Ratel must clear.
- **Random-k** — k tools sampled uniformly from the pool. Floor.
- **BM25 top-k** — lexical retrieval over tool descriptions. Cheap baseline.
- **Embedding top-k** — dense retrieval. Strong, non-trivial baseline. Must use a different embedder from the controller; if that's infeasible, the comparison must be flagged.

If Ratel does not beat embedding top-k on at least one of {accuracy at fixed token budget, tokens at fixed accuracy}, the headline claim is weak.

## Metrics

Per condition × scenario × run:

- **BFCL accuracy**, per-category as defined by BFCL
- **Tokens in** — prompt size delivered to the model
- **Tokens out** — response size
- **Latency** — wall-clock, separated into Ratel-side and model-side
- **Selection precision / recall** (secondary diagnostic) — computed post-hoc from Ratel's trace stream against the scenario's necessary set. `precision = |selected ∩ necessary| / |selected|`; `recall = |selected ∩ necessary| / |necessary|`. Does not move the headline numbers but explains them: a Treatment−Baseline drop with `recall = 1.0` indicates losses outside tool selection (context shape, ordering, over-aggressive trimming); a drop with `recall < 1.0` points to a retrieval gap. Also reported for the BM25 and embedding-top-k baselines, since they are retrievers too.

Headline plots are Pareto frontiers: accuracy vs. tokens-in, and accuracy vs. latency, across N. Single-scalar leaderboard rankings are jointly hard to game.

## Reporting

Every run produces:

- Per-category, per-N accuracy curves with CIs from paired bootstrap
- The three deltas (Oracle−Baseline, Treatment−Baseline, Oracle−Treatment) with CIs
- Pool-composition variance reported separately from run variance
- Selection precision / recall per retrieval-doing condition (Treatment, BM25, embedding top-k), aligned to the same scenarios as the accuracy curves so deltas can be cross-read
- Full configuration manifest: BFCL version, model version, prompts, controller embedder version, candidate corpus version, target (μ, σ), match tolerance, and all seeds

## Open questions

- **Controller embedder choice.** The LLM under test may perceive tool similarity differently from any single embedder. Cross-check by varying the controller embedder across at least two choices and confirming pool composition stays stable. Settle on a default after that data exists.
- **Empirical realism of the synthetic target.** v0 ships with a parametric normal target only — we do not yet know whether our chosen (μ, σ) sits in a realistic region of real tool catalogs. Calibration step: scrape pairwise similarity distributions from real MCP servers (GitHub, Linear, Slack, Stripe, …) and overlay against synthetic. Once those distributions are in hand, decide whether to switch the target from parametric normal to empirical CDF, or keep parametric as the sweep axis with empirical as a fixed reference point. Real MCP tools may also enter the candidate corpus at that point.
- **Distribution-match tolerance.** The K-S threshold below which a sampled pool is accepted as matching the target. Set empirically — tight enough that pools at fixed N are interchangeable, loose enough that sampling at N = 500 from a 2k corpus is feasible.
- **Multi-turn similarity scoping.** Default is union across turns; revisit if per-turn pool construction proves necessary for future categories.
