//! End-to-end checks for the MetaTool ingestion pipeline.
//!
//! Drives `ingest_to_jsonl` against an inline fixture (no on-disk test-data
//! dependency, since the corpus snapshot is no longer committed) and asserts
//! that the produced JSONL parses cleanly through the corpus loader and
//! round-trips through the retrieval runner with finite metrics.

use std::path::Path;

use ratel_benchmark_retrieval::corpus::parse_scenarios;
use ratel_benchmark_retrieval::ingest::metatool::{MetaToolPaths, ingest_to_jsonl};
use ratel_benchmark_retrieval::runner::{RunConfig, run_retrieval};

const PLUGINS_JSON: &str = r#"{
    "WeatherTool": "Get current weather and forecasts.",
    "FinanceTool": "Real-time stock and crypto prices.",
    "NewsTool": "Latest world headlines.",
    "MapTool": "Maps and routing.",
    "TranslateTool": "Language translation.",
    "CalendarTool": "Calendar events and reminders."
}"#;

const SINGLE_TOOL_CSV: &str = "Query,Tool\n\
What's the forecast?,WeatherTool\n\
AAPL stock price?,FinanceTool\n\
Latest news?,NewsTool\n\
Driving directions?,MapTool\n\
Translate hello to French,TranslateTool\n\
Schedule a meeting,CalendarTool\n";

const MULTI_TOOL_JSON: &str = r#"[
    { "query": "weather + headlines", "tool": ["WeatherTool", "NewsTool"] },
    { "query": "tesla news + stock", "tool": ["NewsTool", "FinanceTool"] }
]"#;

fn write_fixture(dir: &Path) -> MetaToolPaths {
    let plugins = dir.join("plugin_des.json");
    let single = dir.join("all_clean_data.csv");
    let multi = dir.join("multi_tool_query_golden.json");
    std::fs::write(&plugins, PLUGINS_JSON).unwrap();
    std::fs::write(&single, SINGLE_TOOL_CSV).unwrap();
    std::fs::write(&multi, MULTI_TOOL_JSON).unwrap();
    MetaToolPaths {
        plugins,
        single_tool: single,
        multi_tool: Some(multi),
    }
}

#[test]
fn ingest_writes_jsonl_parsable_by_corpus_loader() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_fixture(dir.path());
    let out = dir.path().join("out.jsonl");

    let stats = ingest_to_jsonl(&paths, &out).unwrap();
    assert_eq!(stats.plugins_loaded, 6);
    assert_eq!(stats.single_tool_in, 6);
    assert_eq!(stats.multi_tool_in, 2);
    assert_eq!(stats.skipped_unknown_gold, 0);
    // 6 single + 2 multi (tool) + 2 multi (skill) = 10.
    assert_eq!(stats.scenarios_out, 10);
    assert_eq!(stats.skill_scenarios_out, 2);

    let scenarios =
        parse_scenarios(std::io::BufReader::new(std::fs::File::open(&out).unwrap())).unwrap();
    assert_eq!(scenarios.len(), 10);
    for s in &scenarios {
        assert!(!s.gold_tools.is_empty());
        for tool in &s.candidate_pool {
            // MetaTool plugins ship without parameter schemas.
            assert!(tool.input_schema.as_object().is_some_and(|o| o.is_empty()));
            assert!(tool.output_schema.as_object().is_some_and(|o| o.is_empty()));
        }
        assert!(s.judge_criteria.is_none());
        assert!(s.id.starts_with("metatool-"));
        // A scenario is either tool-mode or skill-mode, never both.
        assert!(s.candidate_pool.is_empty() != s.candidate_skills.is_empty());
    }
}

#[test]
fn ingest_round_trips_through_retrieval_runner() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_fixture(dir.path());
    let corpus = dir.path().join("corpus.jsonl");
    ingest_to_jsonl(&paths, &corpus).unwrap();

    let retrieval_out = dir.path().join("retrieval.jsonl");
    let summary_out = dir.path().join("retrieval-summary.jsonl");
    let summary = run_retrieval(&RunConfig {
        corpus_path: corpus.clone(),
        output_path: retrieval_out.clone(),
        summary_path: summary_out.clone(),
        scenario_limit: None,
        top_ks: vec![1, 3],
        pool_sizes: vec![3, 6],
        seed: 42,
    })
    .unwrap();
    assert_eq!(summary.scenarios, 10);
    // 10 scenarios × 2 pool sizes × 2 K cutoffs = 40 rows.
    assert_eq!(summary.rows_written, 40);

    let summary_body = std::fs::read_to_string(&summary_out).unwrap();
    let summary_line = summary_body.lines().next().expect("one summary line");
    let summary_json: serde_json::Value = serde_json::from_str(summary_line).unwrap();
    assert_eq!(summary_json["scenarios"], 10);
    assert_eq!(summary_json["pool_sizes"], serde_json::json!([3, 6]));

    // Three buckets: single/tool, multi/tool, multi/skill — same metric shape.
    let by_bucket = summary_json["by_bucket"].as_array().unwrap();
    assert_eq!(by_bucket.len(), 3);
    let skill = by_bucket
        .iter()
        .find(|b| b["subset"] == "multi-tool" && b["mode"] == "skill")
        .expect("skill bucket present");
    assert_eq!(skill["scenarios"], 2);
    assert_eq!(skill["by_pool_size"].as_array().unwrap().len(), 2);
    assert_eq!(skill["overall"]["by_k"].as_array().unwrap().len(), 2);

    let body = std::fs::read_to_string(&retrieval_out).unwrap();
    let mut row_count = 0usize;
    for line in body.lines().filter(|l| !l.is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        for key in ["recall_at_k", "reciprocal_rank", "precision_at_k"] {
            let v = row[key].as_f64().expect(key);
            assert!(v.is_finite(), "{key} should be finite, got {v}");
            assert!(
                (0.0..=1.0).contains(&v),
                "{key} should be within [0,1], got {v}"
            );
        }
        assert!(row["hit_at_k"].is_boolean());
        row_count += 1;
    }
    assert_eq!(row_count, 40);
}

#[test]
fn ingest_skips_query_with_unknown_gold() {
    let dir = tempfile::tempdir().unwrap();
    let plugins = dir.path().join("plugin_des.json");
    std::fs::write(&plugins, r#"{"WeatherTool":"weather"}"#).unwrap();
    let single = dir.path().join("single.csv");
    std::fs::write(&single, "Query,Tool\nfine,WeatherTool\nbad,Bogus\n").unwrap();

    let out = dir.path().join("out.jsonl");
    let stats = ingest_to_jsonl(
        &MetaToolPaths {
            plugins,
            single_tool: single,
            multi_tool: None,
        },
        &out,
    )
    .unwrap();
    assert_eq!(stats.scenarios_out, 1);
    assert_eq!(stats.skipped_unknown_gold, 1);
}
