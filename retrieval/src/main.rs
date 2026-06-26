use std::path::{Path, PathBuf};
use std::process::Command as ShellCommand;

use anyhow::Context;
use clap::{Parser, Subcommand};
use ratel_benchmark_retrieval::ingest::bfcl::{BfclPaths, ingest_to_jsonl as ingest_bfcl};
use ratel_benchmark_retrieval::ingest::metatool::{
    self as metatool, MetaToolPaths, PLUGIN_DES_URL, SINGLE_TOOL_CSV_URL,
    ingest_to_jsonl as ingest_metatool,
};
use ratel_benchmark_retrieval::ingest::sragents::{
    self as sragents, CORPUS_ZIP_URL as SRAGENTS_CORPUS_ZIP_URL,
    INSTANCE_URLS as SRAGENTS_INSTANCE_URLS, SrAgentsPaths, ingest_to_jsonl as ingest_sragents,
};
use ratel_benchmark_retrieval::ingest::toolret::{
    QUERIES_URLS as TOOLRET_QUERIES_URLS, TOOLS_URLS as TOOLRET_TOOLS_URLS, ToolRetPaths,
    ingest_to_jsonl as ingest_toolret,
};
use ratel_benchmark_retrieval::runner::{RunConfig, run_retrieval};
use ratel_benchmark_retrieval::skill_runner::{SkillRunConfig, run_skill_retrieval};

#[derive(Parser)]
#[command(
    name = "ratel-benchmark-retrieval",
    version,
    about = "Ratel benchmark — retrieval layer (ingest + BM25 metrics)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute BM25 retrieval-only metrics for every scenario in a corpus.
    Retrieval {
        /// Path to the JSONL scenario corpus.
        #[arg(short, long)]
        corpus: PathBuf,
        /// Where to write retrieval.jsonl.
        #[arg(short, long, default_value = "results/retrieval.jsonl")]
        output: PathBuf,
        /// Where to append the aggregate overall-performance summary. Each
        /// run adds one JSON line (existing lines are kept), so repeated
        /// runs accumulate a history of experiments to compare over time.
        /// Defaults to `--output` with its extension replaced by
        /// `-summary.jsonl`.
        #[arg(long)]
        summary_output: Option<PathBuf>,
        /// Limit to first N scenarios (full corpus if omitted).
        #[arg(long)]
        scenarios: Option<usize>,
        /// Top-K cutoffs for recall/precision/MRR, comma-separated. One row
        /// per (scenario, pool_size, k) is emitted from a single ranking pass.
        #[arg(long, value_delimiter = ',', default_values_t = [1usize, 3, 5, 10])]
        top_k: Vec<usize>,
        /// Catalog sizes to evaluate at, comma-separated.
        #[arg(long, value_delimiter = ',', default_values_t = [30usize, 150, 600])]
        pool_sizes: Vec<usize>,
        /// Seed for distractor shuffling.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Compute BM25 skill-retrieval metrics over an authored skill corpus
    /// (SR-Agents). Separate from `retrieval` (tools): the skill catalog is the
    /// BM25 index / distractor universe; each instance is a question + gold
    /// skill ids.
    SkillRetrieval {
        /// Path to the JSONL instances (one `SkillInstance` per line).
        #[arg(long, default_value = "test-data/sragents.jsonl")]
        instances: PathBuf,
        /// Path to the JSONL skill catalog (one `SkillSpec` per line).
        #[arg(long, default_value = "test-data/sragents-skills.jsonl")]
        skills_catalog: PathBuf,
        /// Where to write the per-row retrieval JSONL.
        #[arg(short, long, default_value = "results/sragents-skill-retrieval.jsonl")]
        output: PathBuf,
        /// Where to append the aggregate summary. Defaults to `--output` with
        /// its extension replaced by `-summary.jsonl`.
        #[arg(long)]
        summary_output: Option<PathBuf>,
        /// Sample N instances, stratified across datasets (N/num_datasets each,
        /// drawn at random per `--seed`); full set if omitted. Re-run with the
        /// same N + seed to repeat on the identical questions.
        #[arg(long)]
        scenarios: Option<usize>,
        /// Top-K cutoffs, comma-separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1usize, 3, 5, 10])]
        top_k: Vec<usize>,
        /// Catalog sample sizes to evaluate at, comma-separated.
        #[arg(long, value_delimiter = ',', default_values_t = [30usize, 150, 600])]
        pool_sizes: Vec<usize>,
        /// Seed for distractor shuffling and `--scenarios` stratified sampling.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Convert an external corpus into the harness's normalized JSONL format.
    Ingest {
        #[command(subcommand)]
        source: IngestSource,
    },
}

#[derive(Subcommand)]
enum IngestSource {
    /// MetaTool (HowieHwong/MetaTool, MIT). With `--download` the upstream
    /// sources are pulled (via curl) into `--fixtures-dir` before ingesting,
    /// so a clean clone can produce the corpus in one command. The full
    /// upstream query set is emitted — no sampling.
    Metatool {
        /// Where downloaded source files live, mirroring upstream `dataset/`.
        #[arg(long, default_value = "fixtures/metatool")]
        fixtures_dir: PathBuf,
        /// Pull upstream sources into `--fixtures-dir` before ingesting (uses
        /// the system `curl`). Skip the flag to read pre-existing files.
        #[arg(long, default_value_t = false)]
        download: bool,
        /// Override path to `plugin_des.json`. Defaults under `--fixtures-dir`.
        #[arg(long)]
        plugins: Option<PathBuf>,
        /// Override path to `all_clean_data.csv`. Defaults under `--fixtures-dir`.
        #[arg(long)]
        single_tool: Option<PathBuf>,
        /// Override path to `multi_tool_query_golden.json`. Defaults under
        /// `--fixtures-dir`. Pass an empty string to skip multi-tool ingest.
        #[arg(long)]
        multi_tool: Option<PathBuf>,
        /// Where to write the normalized JSONL corpus.
        #[arg(short, long, default_value = "test-data/metatool.jsonl")]
        output: PathBuf,
    },
    /// ToolRet (mangopy/ToolRet-Tools + ToolRet-Queries, Apache-2.0). With
    /// `--download` the upstream parquet files (3 tool subsets + 35 query
    /// sub-corpora) are pulled into `--fixtures-dir`. No sampling — the full
    /// corpus is normalized; rows with unknown gold tools are skipped.
    Toolret {
        /// Where downloaded parquet files live (`<dir>/tools/*.parquet`,
        /// `<dir>/queries/*.parquet`).
        #[arg(long, default_value = "fixtures/toolret")]
        fixtures_dir: PathBuf,
        /// Pull upstream parquet files into `--fixtures-dir` before ingesting
        /// (uses the system `curl`). Skip the flag to read pre-existing files.
        #[arg(long, default_value_t = false)]
        download: bool,
        /// Where to write the normalized JSONL corpus.
        #[arg(short, long, default_value = "test-data/toolret.jsonl")]
        output: PathBuf,
    },
    /// BFCL (Berkeley Function Calling Leaderboard, gorilla-llm on HuggingFace).
    /// With `--download` the four source files (simple + multiple data, plus
    /// their `possible_answer/` ground truth) are pulled into `--fixtures-dir`.
    /// Emits two normalized corpora: single-tool (simple) and multi-tool
    /// (multiple). Rows with unknown gold / no answer are skipped (counted).
    Bfcl {
        /// Where downloaded source files live (mirrors upstream layout).
        #[arg(long, default_value = "fixtures/bfcl")]
        fixtures_dir: PathBuf,
        /// Pull upstream sources into `--fixtures-dir` before ingesting (uses
        /// the system `curl`). Skip the flag to read pre-existing files.
        #[arg(long, default_value_t = false)]
        download: bool,
        /// Where to write the normalized single-tool (simple) corpus.
        #[arg(long, default_value = "test-data/bfcl-simple.jsonl")]
        simple_output: PathBuf,
        /// Where to write the normalized multi-tool (multiple) corpus.
        #[arg(long, default_value = "test-data/bfcl-multiple.jsonl")]
        multiple_output: PathBuf,
    },
    /// SR-Agents (oneal2000/SR-Agents). With `--download` the upstream skill
    /// corpus zip + six instance files are pulled into `--fixtures-dir` and the
    /// corpus is unzipped before ingesting. Produces a skill catalog JSONL and
    /// an instances JSONL for `skill-retrieval`.
    Sragents {
        /// Where downloaded source files live, mirroring upstream `data/bench/`.
        #[arg(long, default_value = "fixtures/sragents")]
        fixtures_dir: PathBuf,
        /// Pull upstream sources into `--fixtures-dir` (and unzip the corpus)
        /// before ingesting (uses the system `curl`). Skip to read existing files.
        #[arg(long, default_value_t = false)]
        download: bool,
        /// Where to write the normalized skill catalog JSONL.
        #[arg(long, default_value = "test-data/sragents-skills.jsonl")]
        catalog_output: PathBuf,
        /// Where to write the normalized instances JSONL.
        #[arg(long, default_value = "test-data/sragents.jsonl")]
        instances_output: PathBuf,
    },
}

/// Default summary path derived from `--output`: strips a trailing `.jsonl`
/// extension (or any extension) and appends `-summary.jsonl`.
fn default_summary_path(output: &Path) -> PathBuf {
    let stem = match output.extension() {
        Some(_) => output.with_extension(""),
        None => output.to_path_buf(),
    };
    let mut name = stem.into_os_string();
    name.push("-summary.jsonl");
    PathBuf::from(name)
}

fn fetch_via_curl(url: &str, dest: &Path) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let status = ShellCommand::new("curl")
        .args(["-sSL", "--fail", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .with_context(|| format!("invoking curl for {url}"))?;
    if !status.success() {
        anyhow::bail!(
            "curl exited with status {} fetching {url}",
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".into())
        );
    }
    Ok(())
}

fn download_metatool_upstream(paths: &MetaToolPaths) -> anyhow::Result<()> {
    eprintln!("downloading MetaTool upstream sources via curl...");
    fetch_via_curl(PLUGIN_DES_URL, &paths.plugins)?;
    fetch_via_curl(SINGLE_TOOL_CSV_URL, &paths.single_tool)?;
    if let Some(multi) = &paths.multi_tool {
        fetch_via_curl(metatool::MULTI_TOOL_JSON_URL, multi)?;
    }
    Ok(())
}

fn download_toolret_upstream(paths: &ToolRetPaths) -> anyhow::Result<()> {
    eprintln!(
        "downloading ToolRet upstream sources via curl ({} tool + {} query parquet files)...",
        TOOLRET_TOOLS_URLS.len(),
        TOOLRET_QUERIES_URLS.len()
    );
    for ((subset, url), (_, dest)) in TOOLRET_TOOLS_URLS.iter().zip(paths.tools.iter()) {
        eprintln!("  tools/{subset}");
        fetch_via_curl(url, dest)?;
    }
    for ((subset, url), (_, dest)) in TOOLRET_QUERIES_URLS.iter().zip(paths.queries.iter()) {
        eprintln!("  queries/{subset}");
        fetch_via_curl(url, dest)?;
    }
    Ok(())
}

fn download_bfcl_upstream(paths: &BfclPaths) -> anyhow::Result<()> {
    let targets = paths.download_targets();
    eprintln!(
        "downloading BFCL upstream sources via curl ({} files)...",
        targets.len()
    );
    for (url, dest) in targets {
        fetch_via_curl(url, &dest)?;
    }
    Ok(())
}

fn download_sragents_upstream(paths: &SrAgentsPaths) -> anyhow::Result<()> {
    eprintln!(
        "downloading SR-Agents upstream sources via curl (skill corpus zip + {} instance files)...",
        SRAGENTS_INSTANCE_URLS.len()
    );
    eprintln!("  corpus/corpus.json.zip");
    fetch_via_curl(SRAGENTS_CORPUS_ZIP_URL, &paths.corpus_zip)?;
    eprintln!("  unzip → corpus/corpus.json");
    sragents::unzip_corpus(&paths.corpus_zip, &paths.corpus_json)?;
    for ((name, url), (_, dest)) in SRAGENTS_INSTANCE_URLS.iter().zip(paths.instances.iter()) {
        eprintln!("  instances/{name}");
        fetch_via_curl(url, dest)?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Retrieval {
            corpus,
            output,
            summary_output,
            scenarios,
            top_k,
            pool_sizes,
            seed,
        } => {
            let summary_path = summary_output.unwrap_or_else(|| default_summary_path(&output));
            let cfg = RunConfig {
                corpus_path: corpus,
                output_path: output.clone(),
                summary_path,
                scenario_limit: scenarios,
                top_ks: top_k,
                pool_sizes,
                seed,
            };
            let summary = run_retrieval(&cfg)?;
            println!(
                "wrote {} rows for {} scenarios → {}; summary appended → {}",
                summary.rows_written,
                summary.scenarios,
                output.display(),
                summary.summary_path.display()
            );
        }
        Command::SkillRetrieval {
            instances,
            skills_catalog,
            output,
            summary_output,
            scenarios,
            top_k,
            pool_sizes,
            seed,
        } => {
            let summary_path = summary_output.unwrap_or_else(|| default_summary_path(&output));
            let cfg = SkillRunConfig {
                instances_path: instances,
                skills_catalog_path: skills_catalog,
                output_path: output.clone(),
                summary_path,
                scenario_limit: scenarios,
                top_ks: top_k,
                pool_sizes,
                seed,
            };
            let summary = run_skill_retrieval(&cfg)?;
            println!(
                "wrote {} rows for {} skill instances → {}; summary appended → {}",
                summary.rows_written,
                summary.scenarios,
                output.display(),
                summary.summary_path.display()
            );
        }
        Command::Ingest { source } => match source {
            IngestSource::Metatool {
                fixtures_dir,
                download,
                plugins,
                single_tool,
                multi_tool,
                output,
            } => {
                let defaults = MetaToolPaths::under_fixtures_dir(&fixtures_dir);
                let paths = MetaToolPaths {
                    plugins: plugins.unwrap_or(defaults.plugins),
                    single_tool: single_tool.unwrap_or(defaults.single_tool),
                    // An empty `--multi-tool` flag (`--multi-tool ""`) opts out;
                    // omitting the flag falls through to the fixtures-dir default.
                    multi_tool: multi_tool
                        .map(|p| {
                            if p.as_os_str().is_empty() {
                                None
                            } else {
                                Some(p)
                            }
                        })
                        .unwrap_or(defaults.multi_tool),
                };
                if download {
                    download_metatool_upstream(&paths)?;
                }
                let stats = ingest_metatool(&paths, &output)?;
                println!(
                    "metatool: {} plugins, {} single-tool in / {} multi-tool in, \
                     {} skipped (unknown gold) → {} scenarios at {}",
                    stats.plugins_loaded,
                    stats.single_tool_in,
                    stats.multi_tool_in,
                    stats.skipped_unknown_gold,
                    stats.scenarios_out,
                    output.display(),
                );
            }
            IngestSource::Toolret {
                fixtures_dir,
                download,
                output,
            } => {
                let paths = ToolRetPaths::under_fixtures_dir(&fixtures_dir);
                if download {
                    download_toolret_upstream(&paths)?;
                }
                let stats = ingest_toolret(&paths, &output)?;
                println!(
                    "toolret: {} tools, {} queries in, \
                     {} skipped (unknown gold), {} skipped (no positive label) \
                     → {} scenarios at {}",
                    stats.tools_loaded,
                    stats.queries_in,
                    stats.skipped_unknown_gold,
                    stats.skipped_no_positive_label,
                    stats.scenarios_out,
                    output.display(),
                );
            }
            IngestSource::Bfcl {
                fixtures_dir,
                download,
                simple_output,
                multiple_output,
            } => {
                let paths = BfclPaths::under_fixtures_dir(&fixtures_dir);
                if download {
                    download_bfcl_upstream(&paths)?;
                }
                let stats = ingest_bfcl(&paths, &simple_output, &multiple_output)?;
                println!(
                    "bfcl: {} simple in / {} multiple in, \
                     {} skipped (unknown gold), {} skipped (missing answer), \
                     {} ids disambiguated → {} scenarios at {} + {}",
                    stats.simple_in,
                    stats.multiple_in,
                    stats.skipped_unknown_gold,
                    stats.skipped_missing_answer,
                    stats.collisions_disambiguated,
                    stats.scenarios_out,
                    simple_output.display(),
                    multiple_output.display(),
                );
            }
            IngestSource::Sragents {
                fixtures_dir,
                download,
                catalog_output,
                instances_output,
            } => {
                let paths = SrAgentsPaths::under_fixtures_dir(&fixtures_dir);
                if download {
                    download_sragents_upstream(&paths)?;
                }
                let stats = ingest_sragents(&paths, &catalog_output, &instances_output)?;
                let by_dataset = stats
                    .by_dataset
                    .iter()
                    .map(|(d, n)| format!("{d}={n}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "sragents: {} skills → {}; {} instances in, {} skipped (unknown gold) \
                     → {} instances [{}] at {}",
                    stats.skills_loaded,
                    catalog_output.display(),
                    stats.instances_in,
                    stats.skipped_unknown_gold,
                    stats.instances_out,
                    by_dataset,
                    instances_output.display(),
                );
            }
        },
    }
    Ok(())
}
