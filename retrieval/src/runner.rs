//! Drives retrieval-only metrics across every scenario in a corpus.
//!
//! For each scenario, evaluates BM25 quality at each requested catalog scale,
//! using tools from other scenarios as distractors. Emits one JSONL row per
//! `(scenario, pool_size)` cell.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::corpus::{Identified, Scenario, SkillSpec, ToolSpec, load_scenarios};
use crate::retrieval::{RetrievalMetrics, build_pool, evaluate_at_ks, evaluate_skills_at_ks};
use crate::stats::{self, Stats};

/// Inputs for one retrieval-only run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub corpus_path: PathBuf,
    pub output_path: PathBuf,
    /// Where to append the aggregate overall-performance summary. One
    /// compact JSON object per run is appended as a new line (JSONL), so
    /// repeated runs accumulate a history you can compare across time —
    /// distinguished by each line's `generated_at` timestamp. Created if it
    /// doesn't exist yet.
    pub summary_path: PathBuf,
    pub scenario_limit: Option<usize>,
    /// K cutoffs to score at, in the order the runner should emit them.
    /// Each `(scenario, pool_size)` cell produces one JSONL row per K.
    pub top_ks: Vec<usize>,
    pub pool_sizes: Vec<usize>,
    pub seed: u64,
}

/// One row of retrieval-only output. Joined to agent-loop rows by `scenario_id`.
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalRow {
    pub scenario_id: String,
    /// Scenario category (e.g. `metatool-single` / `metatool-multi` /
    /// `metatool-skill`), carried through so the report can bucket rows by
    /// `(subset, mode)` without re-deriving it. `None` for category-less
    /// corpora (the report then falls back to gold-set size).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    pub target_pool_size: usize,
    pub actual_pool_size: usize,
    #[serde(flatten)]
    pub metrics: RetrievalMetrics,
}

/// Aggregation bucket: a `(subset, retrieval mode)` pair. A skill is the
/// multi-tool *skill-retrieval* mode (one gold skill bundle), not a separate
/// subset; single-tool and multi-tool tool-retrieval are the other two.
/// `mode == "skill"` also selects the skill distractor universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bucket {
    pub subset: &'static str,
    pub mode: &'static str,
}

impl Bucket {
    fn is_skill(&self) -> bool {
        self.mode == "skill"
    }
}

/// Map a scenario to its `(subset, mode)` bucket. MetaTool single-tool queries
/// are tool-retrieval; each MetaTool multi-tool query is scored both ways — as
/// individual tools (`metatool-multi`, `ToolRegistry`) and as one skill bundle
/// (`metatool-skill`, `SkillRegistry`). Category-less corpora (e.g. ToolRet)
/// fall back to gold-set size in tool mode.
fn bucket_of(scenario: &Scenario) -> Bucket {
    match scenario.category.as_deref() {
        Some("metatool-single") => Bucket {
            subset: "single-tool",
            mode: "tool",
        },
        Some("metatool-multi") => Bucket {
            subset: "multi-tool",
            mode: "tool",
        },
        Some("metatool-skill") => Bucket {
            subset: "multi-tool",
            mode: "skill",
        },
        _ => {
            if scenario.gold_tools.len() > 1 {
                Bucket {
                    subset: "multi-tool",
                    mode: "tool",
                }
            } else {
                Bucket {
                    subset: "single-tool",
                    mode: "tool",
                }
            }
        }
    }
}

/// Fixed output order for the summary's `by_bucket` blocks. Any bucket not
/// listed here (future categories) is appended after these, sorted, so output
/// stays deterministic.
const BUCKET_ORDER: &[(&str, &str)] = &[
    ("single-tool", "tool"),
    ("multi-tool", "tool"),
    ("multi-tool", "skill"),
];

/// Per-bucket accumulators, mirroring the previous single-bucket set.
#[derive(Debug, Default)]
struct BucketAcc {
    scenarios: usize,
    overall_score: ScoreAcc,
    overall_k: HashMap<usize, KAcc>,
    by_pool_score: HashMap<usize, ScoreAcc>,
    by_pool_k: HashMap<(usize, usize), KAcc>,
}

impl BucketAcc {
    fn into_summary(
        mut self,
        bucket: Bucket,
        top_ks: &[usize],
        pool_sizes: &[usize],
    ) -> BucketSummary {
        let overall = PoolSizeSummary::build(None, self.overall_score, top_ks, self.overall_k);
        let by_pool_size = pool_sizes
            .iter()
            .map(|&pool_size| {
                let score = self.by_pool_score.remove(&pool_size).unwrap_or_default();
                let ks: HashMap<usize, KAcc> = top_ks
                    .iter()
                    .filter_map(|&k| self.by_pool_k.remove(&(pool_size, k)).map(|acc| (k, acc)))
                    .collect();
                PoolSizeSummary::build(Some(pool_size), score, top_ks, ks)
            })
            .collect();
        BucketSummary {
            subset: bucket.subset.to_string(),
            mode: bucket.mode.to_string(),
            scenarios: self.scenarios,
            overall,
            by_pool_size,
        }
    }
}

pub fn run_retrieval(config: &RunConfig) -> anyhow::Result<RunSummary> {
    let scenarios = load_scenarios(&config.corpus_path)?;
    let scenarios: Vec<Scenario> = match config.scenario_limit {
        Some(n) => scenarios.into_iter().take(n).collect(),
        None => scenarios,
    };

    if let Some(parent) = config.output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating output dir {}: {e}", parent.display()))?;
    }
    let file = File::create(&config.output_path)
        .map_err(|e| anyhow::anyhow!("creating {}: {e}", config.output_path.display()))?;
    let mut writer = BufWriter::new(file);

    // Distractors are pooled per universe: tool scenarios compete against the
    // plugin tool universe (real `ToolRegistry`); skill scenarios compete only
    // against other skills (real `SkillRegistry`). Each mode stays honest.
    let tool_distractors = collect_tool_distractors(&scenarios);
    let skill_distractors = collect_skill_distractors(&scenarios);
    let mut rows = 0usize;

    // One accumulator set per `(subset, mode)` bucket.
    let mut buckets: HashMap<Bucket, BucketAcc> = HashMap::new();

    for scenario in &scenarios {
        let bucket = bucket_of(scenario);

        // Build per-pool metrics with the registry that matches the bucket's
        // mode. Both branches yield the same `(actual_pool_size, metrics)` shape.
        let per_pool: Vec<(usize, Vec<RetrievalMetrics>)> = if bucket.is_skill() {
            evaluate_scenario(
                &scenario.candidate_skills,
                &skill_distractors,
                &scenario.id,
                &scenario.prompt,
                &scenario.gold_tools,
                config.seed,
                &config.pool_sizes,
                &config.top_ks,
                evaluate_skills_at_ks,
            )
        } else {
            evaluate_scenario(
                &scenario.candidate_pool,
                &tool_distractors,
                &scenario.id,
                &scenario.prompt,
                &scenario.gold_tools,
                config.seed,
                &config.pool_sizes,
                &config.top_ks,
                evaluate_at_ks,
            )
        };

        let acc = buckets.entry(bucket).or_default();
        acc.scenarios += 1;

        for (&target_size, (actual_pool_size, all_metrics)) in
            config.pool_sizes.iter().zip(per_pool.iter())
        {
            // `gold_score` is identical across every `k` entry for this
            // (scenario, pool) cell — pull it once so it isn't double-counted.
            let gold_score = all_metrics.first().and_then(|m| m.gold_score);
            acc.overall_score.push(gold_score);
            acc.by_pool_score
                .entry(target_size)
                .or_default()
                .push(gold_score);

            for metrics in all_metrics {
                acc.overall_k.entry(metrics.k).or_default().push(metrics);
                acc.by_pool_k
                    .entry((target_size, metrics.k))
                    .or_default()
                    .push(metrics);
            }

            for metrics in all_metrics {
                let row = RetrievalRow {
                    scenario_id: scenario.id.clone(),
                    category: scenario.category.clone(),
                    target_pool_size: target_size,
                    actual_pool_size: *actual_pool_size,
                    metrics: metrics.clone(),
                };
                writeln!(writer, "{}", serde_json::to_string(&row)?)?;
                rows += 1;
            }
        }
    }

    writer.flush()?;

    if let Some(parent) = config.summary_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating summary dir {}: {e}", parent.display()))?;
    }
    // Emit buckets in the canonical order, then any extras (sorted) so output
    // is deterministic regardless of HashMap iteration order.
    let mut by_bucket: Vec<BucketSummary> = Vec::new();
    for &(subset, mode) in BUCKET_ORDER {
        let key = Bucket { subset, mode };
        if let Some(acc) = buckets.remove(&key) {
            by_bucket.push(acc.into_summary(key, &config.top_ks, &config.pool_sizes));
        }
    }
    let mut extras: Vec<(Bucket, BucketAcc)> = buckets.into_iter().collect();
    extras.sort_by(|a, b| (a.0.subset, a.0.mode).cmp(&(b.0.subset, b.0.mode)));
    for (key, acc) in extras {
        by_bucket.push(acc.into_summary(key, &config.top_ks, &config.pool_sizes));
    }

    let summary = OverallSummary {
        generated_at: chrono::Utc::now().to_rfc3339(),
        corpus: config.corpus_path.display().to_string(),
        output: config.output_path.display().to_string(),
        scenarios: scenarios.len(),
        rows_written: rows,
        top_k: config.top_ks.clone(),
        pool_sizes: config.pool_sizes.clone(),
        seed: config.seed,
        by_bucket,
    };
    // Append (not overwrite): each run adds one line, so the file accumulates
    // a comparable history across experiments run at different times.
    let mut summary_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.summary_path)
        .map_err(|e| anyhow::anyhow!("opening {}: {e}", config.summary_path.display()))?;
    writeln!(summary_file, "{}", serde_json::to_string(&summary)?)?;

    Ok(RunSummary {
        scenarios: scenarios.len(),
        rows_written: rows,
        summary_path: config.summary_path.clone(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub scenarios: usize,
    pub rows_written: usize,
    pub summary_path: PathBuf,
}

/// Accumulates rank-dependent metrics for one `k` cutoff across scenarios.
#[derive(Debug, Default)]
struct KAcc {
    recall: Vec<f64>,
    precision: Vec<f64>,
    ndcg: Vec<f64>,
    mrr: Vec<f64>,
    hits: usize,
}

impl KAcc {
    fn push(&mut self, m: &RetrievalMetrics) {
        self.recall.push(m.recall_at_k);
        self.precision.push(m.precision_at_k);
        self.ndcg.push(m.ndcg_at_k);
        self.mrr.push(m.reciprocal_rank);
        if m.hit_at_k {
            self.hits += 1;
        }
    }

    fn summarize(&self, k: usize) -> KSummary {
        let n = self.recall.len();
        KSummary {
            k,
            n,
            mean_precision: stats::mean(&self.precision),
            median_precision: stats::median(&self.precision),
            mean_recall: stats::mean(&self.recall),
            median_recall: stats::median(&self.recall),
            mean_ndcg: stats::mean(&self.ndcg),
            median_ndcg: stats::median(&self.ndcg),
            mean_mrr: stats::mean(&self.mrr),
            median_mrr: stats::median(&self.mrr),
            hit_rate: if n == 0 {
                0.0
            } else {
                self.hits as f64 / n as f64
            },
        }
    }
}

/// Accumulates BM25 gold-tool scores across scenarios for one group
/// (overall, or one pool size). `total` is the number of scenarios
/// evaluated in the group; `scores` holds only the `Some` values, so
/// `coverage = scores.len() / total`.
#[derive(Debug, Default)]
struct ScoreAcc {
    total: usize,
    scores: Vec<f64>,
}

impl ScoreAcc {
    fn push(&mut self, gold_score: Option<f64>) {
        self.total += 1;
        if let Some(s) = gold_score {
            self.scores.push(s);
        }
    }

    fn summarize(&self) -> GoldScoreSummary {
        GoldScoreSummary {
            stats: stats::summarize(&self.scores),
            n: self.scores.len(),
            coverage: if self.total == 0 {
                0.0
            } else {
                self.scores.len() as f64 / self.total as f64
            },
        }
    }
}

/// BM25 score for the gold tool, summarized over scenarios where it was
/// found (`n` of `total` — see `coverage`).
#[derive(Debug, Clone, Serialize)]
pub struct GoldScoreSummary {
    #[serde(flatten)]
    pub stats: Stats,
    pub n: usize,
    pub coverage: f64,
}

/// Mean/median rank-quality metrics for one `k` cutoff, aggregated across
/// every scenario in the group.
#[derive(Debug, Clone, Serialize)]
pub struct KSummary {
    pub k: usize,
    pub n: usize,
    pub mean_precision: f64,
    pub median_precision: f64,
    pub mean_recall: f64,
    pub median_recall: f64,
    pub mean_ndcg: f64,
    pub median_ndcg: f64,
    pub mean_mrr: f64,
    pub median_mrr: f64,
    pub hit_rate: f64,
}

/// One aggregation group: either "overall" (`pool_size: None`, spans every
/// pool size) or one specific pool size.
#[derive(Debug, Clone, Serialize)]
pub struct PoolSizeSummary {
    pub pool_size: Option<usize>,
    pub n: usize,
    pub bm25_gold_score: GoldScoreSummary,
    pub by_k: Vec<KSummary>,
}

impl PoolSizeSummary {
    fn build(
        pool_size: Option<usize>,
        score: ScoreAcc,
        top_ks: &[usize],
        mut k_accs: HashMap<usize, KAcc>,
    ) -> Self {
        let n = score.total;
        let by_k = top_ks
            .iter()
            .filter_map(|&k| k_accs.remove(&k).map(|acc| acc.summarize(k)))
            .collect();
        PoolSizeSummary {
            pool_size,
            n,
            bm25_gold_score: score.summarize(),
            by_k,
        }
    }
}

/// Per-bucket metrics block: the same `overall` + `by_pool_size` shape as
/// before, computed separately for one `(subset, mode)` pair. A run emits one
/// block per bucket — `{single-tool, tool}`, `{multi-tool, tool}`, and
/// `{multi-tool, skill}` for MetaTool.
#[derive(Debug, Clone, Serialize)]
pub struct BucketSummary {
    /// `single-tool` | `multi-tool`.
    pub subset: String,
    /// `tool` | `skill` (skill is the multi-tool skill-retrieval mode).
    pub mode: String,
    /// Distinct scenarios in this bucket.
    pub scenarios: usize,
    pub overall: PoolSizeSummary,
    pub by_pool_size: Vec<PoolSizeSummary>,
}

/// Overall-performance summary written alongside the per-row JSONL. Metrics are
/// split into one `by_bucket` block per `(subset, mode)` so single-tool,
/// multi-tool (tool retrieval), and multi-tool (skill retrieval) are reported
/// separately with the same metric set.
#[derive(Debug, Clone, Serialize)]
pub struct OverallSummary {
    pub generated_at: String,
    pub corpus: String,
    pub output: String,
    pub scenarios: usize,
    pub rows_written: usize,
    pub top_k: Vec<usize>,
    pub pool_sizes: Vec<usize>,
    pub seed: u64,
    pub by_bucket: Vec<BucketSummary>,
}

/// Pool every tool-mode scenario's candidate tools into the tool distractor
/// universe (deduped by id). Each scenario later drops entries already in its
/// own pool.
fn collect_tool_distractors(scenarios: &[Scenario]) -> Vec<ToolSpec> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<ToolSpec> = Vec::new();
    for s in scenarios {
        if bucket_of(s).is_skill() {
            continue;
        }
        for t in &s.candidate_pool {
            if seen.insert(t.id.clone()) {
                out.push(t.clone());
            }
        }
    }
    out
}

/// Pool every skill-mode scenario's candidate skills into the skill distractor
/// universe (deduped by id) — skills only ever compete against other skills.
fn collect_skill_distractors(scenarios: &[Scenario]) -> Vec<SkillSpec> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<SkillSpec> = Vec::new();
    for s in scenarios {
        if !bucket_of(s).is_skill() {
            continue;
        }
        for sk in &s.candidate_skills {
            if seen.insert(sk.id.clone()) {
                out.push(sk.clone());
            }
        }
    }
    out
}

/// Build per-pool-size metrics for one scenario, generic over tools vs skills.
/// Filters the scenario's own candidates out of `universe`, shuffles
/// deterministically, then evaluates at each pool size with `evaluate`. Returns
/// one `(actual_pool_size, metrics)` per requested pool size, in order.
#[allow(clippy::too_many_arguments)]
fn evaluate_scenario<T: Identified + Clone>(
    scenario_pool: &[T],
    universe: &[T],
    scenario_id: &str,
    prompt: &str,
    gold: &[String],
    seed: u64,
    pool_sizes: &[usize],
    top_ks: &[usize],
    evaluate: impl Fn(&[T], &str, &[String], &[usize]) -> Vec<RetrievalMetrics>,
) -> Vec<(usize, Vec<RetrievalMetrics>)> {
    let own: HashSet<&str> = scenario_pool.iter().map(|t| t.id()).collect();
    let mut distractors: Vec<T> = universe
        .iter()
        .filter(|t| !own.contains(t.id()))
        .cloned()
        .collect();
    // Per-scenario shuffle for deterministic-but-varied distractor ordering.
    let mut rng = scenario_rng(scenario_id, seed);
    distractors.shuffle(&mut rng);

    pool_sizes
        .iter()
        .map(|&size| {
            let pool = build_pool(scenario_pool, &distractors, size);
            let metrics = evaluate(&pool, prompt, gold, top_ks);
            (pool.len(), metrics)
        })
        .collect()
}

fn scenario_rng(scenario_id: &str, seed: u64) -> rand::rngs::StdRng {
    // Mix the scenario id into the seed so distractor orderings vary per scenario
    // but stay reproducible for the same (id, seed) pair.
    let mut h: u64 = seed;
    for b in scenario_id.bytes() {
        h = h.wrapping_mul(1099511628211).wrapping_add(b as u64);
    }
    rand::rngs::StdRng::seed_from_u64(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::ToolSpec;
    use serde_json::json;
    use std::io::Read;

    fn t(id: &str, desc: &str) -> ToolSpec {
        ToolSpec {
            id: id.into(),
            name: id.into(),
            description: desc.into(),
            input_schema: json!({}),
            output_schema: json!({}),
        }
    }

    fn write_corpus(scenarios: &[Scenario]) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut writer = std::io::BufWriter::new(f.as_file());
        for s in scenarios {
            writeln!(writer, "{}", serde_json::to_string(s).unwrap()).unwrap();
        }
        writer.flush().unwrap();
        drop(writer);
        f
    }

    fn scenario(id: &str, prompt: &str, pool: Vec<ToolSpec>, gold: &[&str]) -> Scenario {
        Scenario {
            id: id.into(),
            prompt: prompt.into(),
            candidate_pool: pool,
            candidate_skills: vec![],
            gold_tools: gold.iter().map(|s| (*s).to_string()).collect(),
            judge_criteria: None,
            category: None,
        }
    }

    #[test]
    fn run_emits_one_row_per_scenario_pool_and_k() {
        let scenarios = vec![
            scenario(
                "s1",
                "read a file from disk",
                vec![t("fs.read", "Read a file from disk.")],
                &["fs.read"],
            ),
            scenario(
                "s2",
                "send an email to a recipient",
                vec![t("mail.send", "Send an email to a recipient.")],
                &["mail.send"],
            ),
        ];
        let corpus = write_corpus(&scenarios);
        let out = tempfile::NamedTempFile::new().unwrap();
        let summary_out = tempfile::NamedTempFile::new().unwrap();
        let cfg = RunConfig {
            corpus_path: corpus.path().to_path_buf(),
            output_path: out.path().to_path_buf(),
            summary_path: summary_out.path().to_path_buf(),
            scenario_limit: None,
            top_ks: vec![1, 3],
            pool_sizes: vec![1, 2, 5],
            seed: 42,
        };
        let summary = run_retrieval(&cfg).unwrap();
        assert_eq!(summary.scenarios, 2);
        // 2 scenarios × 3 pool sizes × 2 K values = 12 rows.
        assert_eq!(summary.rows_written, 12);

        let mut contents = String::new();
        std::fs::File::open(out.path())
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 12);
        let mut ks_seen = std::collections::HashSet::new();
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            ks_seen.insert(v["k"].as_u64().unwrap() as usize);
        }
        assert_eq!(ks_seen, std::collections::HashSet::from([1, 3]));
    }

    #[test]
    fn run_respects_scenario_limit() {
        let scenarios = vec![
            scenario("s1", "p1", vec![t("a", "alpha")], &["a"]),
            scenario("s2", "p2", vec![t("b", "beta")], &["b"]),
            scenario("s3", "p3", vec![t("c", "gamma")], &["c"]),
        ];
        let corpus = write_corpus(&scenarios);
        let out = tempfile::NamedTempFile::new().unwrap();
        let summary_out = tempfile::NamedTempFile::new().unwrap();
        let cfg = RunConfig {
            corpus_path: corpus.path().to_path_buf(),
            output_path: out.path().to_path_buf(),
            summary_path: summary_out.path().to_path_buf(),
            scenario_limit: Some(2),
            top_ks: vec![3],
            pool_sizes: vec![5],
            seed: 42,
        };
        let summary = run_retrieval(&cfg).unwrap();
        assert_eq!(summary.scenarios, 2);
        assert_eq!(summary.rows_written, 2);
    }

    #[test]
    fn run_creates_missing_output_directory() {
        let scenarios = vec![scenario("s1", "p1", vec![t("a", "alpha")], &["a"])];
        let corpus = write_corpus(&scenarios);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("nested/sub/output.jsonl");
        let summary_out = dir.path().join("nested/sub/output-summary.jsonl");
        let cfg = RunConfig {
            corpus_path: corpus.path().to_path_buf(),
            output_path: out.clone(),
            summary_path: summary_out.clone(),
            scenario_limit: None,
            top_ks: vec![3],
            pool_sizes: vec![1],
            seed: 42,
        };
        run_retrieval(&cfg).unwrap();
        assert!(out.exists());
        assert!(summary_out.exists());
    }

    #[test]
    fn summary_line_has_overall_and_per_pool_breakdown() {
        let scenarios = vec![
            scenario(
                "s1",
                "read a file from disk",
                vec![t("fs.read", "Read a file from disk.")],
                &["fs.read"],
            ),
            scenario(
                "s2",
                "send an email to a recipient",
                vec![t("mail.send", "Send an email to a recipient.")],
                &["mail.send"],
            ),
        ];
        let corpus = write_corpus(&scenarios);
        let out = tempfile::NamedTempFile::new().unwrap();
        let summary_out = tempfile::NamedTempFile::new().unwrap();
        let cfg = RunConfig {
            corpus_path: corpus.path().to_path_buf(),
            output_path: out.path().to_path_buf(),
            summary_path: summary_out.path().to_path_buf(),
            scenario_limit: None,
            top_ks: vec![1, 3],
            pool_sizes: vec![1, 5],
            seed: 42,
        };
        let run_summary = run_retrieval(&cfg).unwrap();
        assert_eq!(run_summary.summary_path, summary_out.path());

        let mut contents = String::new();
        std::fs::File::open(summary_out.path())
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1, "one summary line per run");
        let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();

        assert_eq!(v["scenarios"], 2);
        assert_eq!(v["rows_written"], 8); // 2 scenarios × 2 pools × 2 k's
        assert_eq!(v["pool_sizes"], serde_json::json!([1, 5]));

        // Both scenarios have one gold tool and no category → single-tool/tool.
        let by_bucket = v["by_bucket"].as_array().unwrap();
        assert_eq!(by_bucket.len(), 1);
        let b = &by_bucket[0];
        assert_eq!(b["subset"], "single-tool");
        assert_eq!(b["mode"], "tool");
        assert_eq!(b["scenarios"], 2);

        assert!(b["overall"]["pool_size"].is_null());
        assert_eq!(b["overall"]["n"], 4); // 2 scenarios × 2 pools
        let overall_by_k = b["overall"]["by_k"].as_array().unwrap();
        assert_eq!(overall_by_k.len(), 2);
        assert_eq!(overall_by_k[0]["k"], 1);
        assert_eq!(overall_by_k[1]["k"], 3);
        assert!(
            b["overall"]["bm25_gold_score"]["coverage"]
                .as_f64()
                .unwrap()
                > 0.0
        );

        let by_pool = b["by_pool_size"].as_array().unwrap();
        assert_eq!(by_pool.len(), 2);
        assert_eq!(by_pool[0]["pool_size"], 1);
        assert_eq!(by_pool[0]["n"], 2);
        assert_eq!(by_pool[0]["by_k"].as_array().unwrap().len(), 2);
        assert_eq!(by_pool[1]["pool_size"], 5);
    }

    fn categorized(
        id: &str,
        prompt: &str,
        pool: Vec<ToolSpec>,
        gold: &[&str],
        category: &str,
    ) -> Scenario {
        Scenario {
            id: id.into(),
            prompt: prompt.into(),
            candidate_pool: pool,
            candidate_skills: vec![],
            gold_tools: gold.iter().map(|s| (*s).to_string()).collect(),
            judge_criteria: None,
            category: Some(category.into()),
        }
    }

    fn skill(id: &str, description: &str, tags: &[&str], tools: &[&str]) -> SkillSpec {
        SkillSpec {
            id: id.into(),
            name: id.into(),
            description: description.into(),
            tags: tags.iter().map(|s| (*s).to_string()).collect(),
            tools: tools.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn skill_scenario(id: &str, prompt: &str, skill: SkillSpec) -> Scenario {
        Scenario {
            id: id.into(),
            prompt: prompt.into(),
            candidate_pool: vec![],
            candidate_skills: vec![skill],
            gold_tools: vec![id.into()],
            judge_criteria: None,
            category: Some("metatool-skill".into()),
        }
    }

    #[test]
    fn summary_splits_into_single_multi_and_skill_buckets() {
        let scenarios = vec![
            categorized(
                "metatool-st-2",
                "read a file from disk",
                vec![t("fs.read", "Read a file from disk.")],
                &["fs.read"],
                "metatool-single",
            ),
            categorized(
                "metatool-mt-0",
                "read and send a file",
                vec![
                    t("fs.read", "Read a file from disk."),
                    t("mail.send", "Send an email to a recipient."),
                ],
                &["fs.read", "mail.send"],
                "metatool-multi",
            ),
            skill_scenario(
                "metatool-skill-0",
                "read and send a file",
                skill(
                    "metatool-skill-0",
                    "Read a file from disk. Send an email to a recipient.",
                    &["fs.read", "mail.send"],
                    &["fs.read", "mail.send"],
                ),
            ),
        ];
        let corpus = write_corpus(&scenarios);
        let out = tempfile::NamedTempFile::new().unwrap();
        let summary_out = tempfile::NamedTempFile::new().unwrap();
        let cfg = RunConfig {
            corpus_path: corpus.path().to_path_buf(),
            output_path: out.path().to_path_buf(),
            summary_path: summary_out.path().to_path_buf(),
            scenario_limit: None,
            top_ks: vec![1, 3],
            pool_sizes: vec![5],
            seed: 42,
        };
        run_retrieval(&cfg).unwrap();

        let contents = std::fs::read_to_string(summary_out.path()).unwrap();
        let v: serde_json::Value = serde_json::from_str(contents.lines().next().unwrap()).unwrap();
        let by_bucket = v["by_bucket"].as_array().unwrap();
        // Canonical order: single/tool, multi/tool, multi/skill.
        assert_eq!(by_bucket.len(), 3);
        assert_eq!(by_bucket[0]["subset"], "single-tool");
        assert_eq!(by_bucket[0]["mode"], "tool");
        assert_eq!(by_bucket[1]["subset"], "multi-tool");
        assert_eq!(by_bucket[1]["mode"], "tool");
        assert_eq!(by_bucket[2]["subset"], "multi-tool");
        assert_eq!(by_bucket[2]["mode"], "skill");
        for b in by_bucket {
            assert_eq!(b["scenarios"], 1);
        }

        // Per-row category is carried through for the report layer.
        let rows = std::fs::read_to_string(out.path()).unwrap();
        let skill_row = rows
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .find(|r| r["scenario_id"] == "metatool-skill-0")
            .unwrap();
        assert_eq!(skill_row["category"], "metatool-skill");
    }

    #[test]
    fn skill_scenario_is_evaluated_via_skill_registry() {
        // A multi-tool scenario carries its bundle in candidate_skills (not
        // candidate_pool) and is scored against the skill universe via the real
        // SkillRegistry. With one skill and no others, it's the only candidate,
        // so it ranks first → hit@1, and actual_pool_size is 1.
        let scenarios = vec![skill_scenario(
            "metatool-mt-0",
            "send an email to a recipient",
            skill(
                "metatool-mt-0",
                "Send an email via SMTP to a recipient.",
                &["mail.send"],
                &["mail.send"],
            ),
        )];
        let corpus = write_corpus(&scenarios);
        let out = tempfile::NamedTempFile::new().unwrap();
        let summary_out = tempfile::NamedTempFile::new().unwrap();
        let cfg = RunConfig {
            corpus_path: corpus.path().to_path_buf(),
            output_path: out.path().to_path_buf(),
            summary_path: summary_out.path().to_path_buf(),
            scenario_limit: None,
            top_ks: vec![1],
            pool_sizes: vec![50],
            seed: 42,
        };
        run_retrieval(&cfg).unwrap();

        let rows = std::fs::read_to_string(out.path()).unwrap();
        let r: serde_json::Value = serde_json::from_str(rows.lines().next().unwrap()).unwrap();
        assert_eq!(r["category"], "metatool-skill");
        // Skill universe has only this one skill → no distractors added.
        assert_eq!(r["actual_pool_size"], 1);
        assert_eq!(r["hit_at_k"], true);
        assert_eq!(r["recall_at_k"], 1.0);
    }

    #[test]
    fn summary_jsonl_accumulates_one_line_per_run() {
        let scenarios = vec![scenario(
            "s1",
            "read a file from disk",
            vec![t("fs.read", "Read a file from disk.")],
            &["fs.read"],
        )];
        let corpus = write_corpus(&scenarios);
        let dir = tempfile::tempdir().unwrap();
        let summary_out = dir.path().join("summary.jsonl");

        for run in 0..3 {
            let out = dir.path().join(format!("retrieval-{run}.jsonl"));
            let cfg = RunConfig {
                corpus_path: corpus.path().to_path_buf(),
                output_path: out,
                summary_path: summary_out.clone(),
                scenario_limit: None,
                top_ks: vec![1],
                pool_sizes: vec![1],
                seed: 42,
            };
            run_retrieval(&cfg).unwrap();
        }

        let contents = std::fs::read_to_string(&summary_out).unwrap();
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 3, "each run appends one line, none overwrite");
        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["scenarios"], 1);
        }
    }

    #[test]
    fn distractor_shuffle_is_deterministic_per_seed() {
        let mut a = scenario_rng("s1", 42);
        let mut b = scenario_rng("s1", 42);
        let mut data_a: Vec<u32> = (0..10).collect();
        let mut data_b: Vec<u32> = (0..10).collect();
        data_a.shuffle(&mut a);
        data_b.shuffle(&mut b);
        assert_eq!(data_a, data_b);
    }

    #[test]
    fn distractor_shuffle_varies_per_scenario_id() {
        let mut a = scenario_rng("s1", 42);
        let mut b = scenario_rng("s2", 42);
        let mut data_a: Vec<u32> = (0..50).collect();
        let mut data_b: Vec<u32> = (0..50).collect();
        data_a.shuffle(&mut a);
        data_b.shuffle(&mut b);
        assert_ne!(
            data_a, data_b,
            "different scenario ids should yield different shuffles"
        );
    }
}
