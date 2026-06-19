//! BFCL → normalized [`crate::corpus::Scenario`] JSONL adapter.
//!
//! Source: <https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard>.
//!
//! Upstream layout (JSON **Lines** — one object per line, NOT a JSON array):
//! - `BFCL_v3_simple.json` — one inline `function` per row; model makes one call.
//! - `BFCL_v3_multiple.json` — 2–4 inline `function`s per row; model picks one.
//! - `possible_answer/BFCL_v3_{simple,multiple}.json` — `{ id, ground_truth: [{<fn>: {...}}] }`.
//!
//! Mapping rules:
//! - `prompt` = the user message(s) of the first question turn.
//! - `candidate_pool` = every inline `function` → [`ToolSpec`] with `id == name ==`
//!   the raw BFCL function name (matches the MetaTool convention).
//! - `gold_tools` = the function name(s) keyed in `ground_truth` (one per row for
//!   both subsets). Rows whose gold fn isn't among the inline functions are skipped;
//!   rows with no matching `possible_answer` entry are skipped (both counted).
//! - `category` = `bfcl-simple` / `bfcl-multiple`; scenario id = `bfcl-{subset}-<row.id>`.
//!
//! Two compatibility fixes happen here so the corpus is safe for the strict agent
//! layer (the retrieval layer ignores `input_schema`):
//! - **Schema normalization** ([`normalize_bfcl_schema`]): BFCL's type vocabulary
//!   (`dict`/`float`/`tuple`/`any`/…) is rewritten to valid JSON Schema so provider
//!   tool definitions are well-formed.
//! - **Post-sanitization de-collision** ([`disambiguate_ids`]): the agent sanitizes
//!   tool ids to `^[a-zA-Z0-9_-]+$` and throws if two distinct ids collapse to the
//!   same token. We detect that across the per-file tool universe and rename the
//!   later id deterministically so a full-universe pool can never abort mid-run.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;
use serde_json::Value;

use crate::corpus::{Scenario, ToolSpec};

/// Canonical `resolve/main` URLs for the four BFCL source files.
pub const SIMPLE_DATA_URL: &str = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_simple.json";
pub const MULTIPLE_DATA_URL: &str = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_multiple.json";
pub const SIMPLE_ANSWER_URL: &str = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_simple.json";
pub const MULTIPLE_ANSWER_URL: &str = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_multiple.json";

/// Paths to the upstream BFCL files (data + ground-truth answers).
#[derive(Debug, Clone)]
pub struct BfclPaths {
    pub simple_data: PathBuf,
    pub simple_answer: PathBuf,
    pub multiple_data: Option<PathBuf>,
    pub multiple_answer: Option<PathBuf>,
}

impl BfclPaths {
    /// Default paths under a fixtures directory, mirroring the upstream layout
    /// (`possible_answer/` subdir for the ground-truth files).
    pub fn under_fixtures_dir(fixtures_dir: &Path) -> Self {
        Self {
            simple_data: fixtures_dir.join("BFCL_v3_simple.json"),
            simple_answer: fixtures_dir.join("possible_answer/BFCL_v3_simple.json"),
            multiple_data: Some(fixtures_dir.join("BFCL_v3_multiple.json")),
            multiple_answer: Some(fixtures_dir.join("possible_answer/BFCL_v3_multiple.json")),
        }
    }

    /// The (url, dest) download pairs for whichever files this config references.
    pub fn download_targets(&self) -> Vec<(&'static str, PathBuf)> {
        let mut out = vec![
            (SIMPLE_DATA_URL, self.simple_data.clone()),
            (SIMPLE_ANSWER_URL, self.simple_answer.clone()),
        ];
        if let Some(p) = &self.multiple_data {
            out.push((MULTIPLE_DATA_URL, p.clone()));
        }
        if let Some(p) = &self.multiple_answer {
            out.push((MULTIPLE_ANSWER_URL, p.clone()));
        }
        out
    }
}

/// Counters for one ingestion run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestStats {
    pub simple_in: usize,
    pub multiple_in: usize,
    pub scenarios_out: usize,
    /// Data rows whose gold fn name wasn't among the inline `function`s.
    pub skipped_unknown_gold: usize,
    /// Data rows with no matching `possible_answer` entry.
    pub skipped_missing_answer: usize,
    /// Distinct tool ids renamed because they collided after sanitization.
    pub collisions_disambiguated: usize,
}

#[derive(Debug, Deserialize)]
struct RawBfclRow {
    id: String,
    question: Vec<Vec<RawMessage>>,
    function: Vec<RawFunction>,
}

#[derive(Debug, Deserialize)]
struct RawMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct RawFunction {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct RawAnswerRow {
    id: String,
    /// Each element is a one-key map `{ fn_name: { arg: [acceptable...] } }`.
    ground_truth: Vec<serde_json::Map<String, Value>>,
}

/// Read BFCL inputs and write normalized JSONL: simple → `simple_output`,
/// multiple → `multiple_output` (skipped when its paths are `None`). Two files
/// keep the per-subset distractor universes separate.
pub fn ingest_to_jsonl(
    paths: &BfclPaths,
    simple_output: &Path,
    multiple_output: &Path,
) -> anyhow::Result<IngestStats> {
    let mut stats = IngestStats::default();

    let (mut simple, s_in, s_unknown, s_missing) =
        build_corpus(&paths.simple_data, &paths.simple_answer, "bfcl-simple")?;
    stats.simple_in = s_in;
    stats.skipped_unknown_gold += s_unknown;
    stats.skipped_missing_answer += s_missing;
    stats.collisions_disambiguated += disambiguate_ids(&mut simple);
    write_scenarios(simple_output, &simple)?;

    if let (Some(data), Some(answer)) = (&paths.multiple_data, &paths.multiple_answer) {
        let (mut multiple, m_in, m_unknown, m_missing) =
            build_corpus(data, answer, "bfcl-multiple")?;
        stats.multiple_in = m_in;
        stats.skipped_unknown_gold += m_unknown;
        stats.skipped_missing_answer += m_missing;
        stats.collisions_disambiguated += disambiguate_ids(&mut multiple);
        write_scenarios(multiple_output, &multiple)?;
        stats.scenarios_out = simple.len() + multiple.len();
    } else {
        stats.scenarios_out = simple.len();
    }

    Ok(stats)
}

/// Build the scenarios for one subset file. Returns
/// `(scenarios, rows_in, skipped_unknown_gold, skipped_missing_answer)`.
fn build_corpus(
    data_path: &Path,
    answer_path: &Path,
    id_prefix: &str,
) -> anyhow::Result<(Vec<Scenario>, usize, usize, usize)> {
    let data_file =
        File::open(data_path).with_context(|| format!("opening {}", data_path.display()))?;
    let rows = parse_data(BufReader::new(data_file))
        .with_context(|| format!("parsing {}", data_path.display()))?;

    let answer_file =
        File::open(answer_path).with_context(|| format!("opening {}", answer_path.display()))?;
    let answers = parse_answers(BufReader::new(answer_file))
        .with_context(|| format!("parsing {}", answer_path.display()))?;

    let category = id_prefix.to_string();
    let rows_in = rows.len();
    let mut skipped_unknown_gold = 0usize;
    let mut skipped_missing_answer = 0usize;
    let mut out: Vec<Scenario> = Vec::with_capacity(rows_in);

    for row in &rows {
        let Some(gold) = answers.get(&row.id) else {
            skipped_missing_answer += 1;
            continue;
        };
        match build_scenario(row, gold, id_prefix, &category) {
            Some(s) => out.push(s),
            None => skipped_unknown_gold += 1,
        }
    }

    // Stable id-sorted output keeps re-ingest diffs readable.
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((out, rows_in, skipped_unknown_gold, skipped_missing_answer))
}

fn parse_data<R: BufRead>(reader: R) -> anyhow::Result<Vec<RawBfclRow>> {
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: RawBfclRow = serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("data row at line {}: {e}", idx + 1))?;
        out.push(row);
    }
    Ok(out)
}

/// Parse the ground-truth JSONL into `id → gold fn name(s)` (the keys of each
/// `ground_truth` element).
fn parse_answers<R: BufRead>(reader: R) -> anyhow::Result<HashMap<String, Vec<String>>> {
    let mut out = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: RawAnswerRow = serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("answer row at line {}: {e}", idx + 1))?;
        let gold: Vec<String> = row
            .ground_truth
            .iter()
            .flat_map(|m| m.keys().cloned())
            .collect();
        out.insert(row.id, gold);
    }
    Ok(out)
}

fn build_scenario(
    row: &RawBfclRow,
    gold: &[String],
    id_prefix: &str,
    category: &str,
) -> Option<Scenario> {
    if gold.is_empty() {
        return None;
    }
    let fn_names: HashSet<&str> = row.function.iter().map(|f| f.name.as_str()).collect();
    // Drop the whole row if any gold fn isn't actually offered (mirrors MetaTool).
    if !gold.iter().all(|g| fn_names.contains(g.as_str())) {
        return None;
    }
    let candidate_pool: Vec<ToolSpec> = row
        .function
        .iter()
        .map(|f| ToolSpec {
            id: f.name.clone(),
            name: f.name.clone(),
            description: f.description.trim().to_string(),
            input_schema: normalize_bfcl_schema(f.parameters.clone()),
            output_schema: Value::Object(Default::default()),
        })
        .collect();
    Some(Scenario {
        id: format!("{id_prefix}-{}", row.id),
        prompt: flatten_prompt(&row.question),
        candidate_pool,
        gold_tools: gold.to_vec(),
        judge_criteria: None,
        category: Some(category.to_string()),
    })
}

/// Flatten the nested `question` turns into a single prompt: the user messages
/// joined by newlines (BFCL simple/multiple are single-turn user asks). Falls
/// back to every message's content if no `user` role is present.
fn flatten_prompt(question: &[Vec<RawMessage>]) -> String {
    let user: Vec<&str> = question
        .iter()
        .flatten()
        .filter(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .collect();
    if !user.is_empty() {
        return user.join("\n");
    }
    question
        .iter()
        .flatten()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Rewrite BFCL's parameter `type` vocabulary into valid JSON Schema, recursing
/// into `properties` and array `items`. Unknown/`any` types drop the `type` key
/// (= "anything"). Other keys (`description`, `required`, `enum`, `default`, …)
/// are preserved verbatim.
pub fn normalize_bfcl_schema(value: Value) -> Value {
    match value {
        Value::Object(mut map) => {
            if let Some(Value::String(t)) = map.get("type") {
                match map_type(t) {
                    Some(json_type) => {
                        map.insert("type".into(), Value::String(json_type.into()));
                    }
                    None => {
                        map.remove("type");
                    }
                }
            }
            if let Some(Value::Object(props)) = map.get_mut("properties") {
                for slot in props.values_mut() {
                    *slot = normalize_bfcl_schema(slot.take());
                }
            }
            if let Some(items) = map.get_mut("items") {
                *items = normalize_bfcl_schema(items.take());
            }
            Value::Object(map)
        }
        other => other,
    }
}

/// Map a BFCL type token to its JSON Schema equivalent. `None` means "no valid
/// JSON Schema type" → caller drops the `type` key.
fn map_type(t: &str) -> Option<&'static str> {
    match t {
        "dict" => Some("object"),
        "float" => Some("number"),
        "tuple" => Some("array"),
        "integer" => Some("integer"),
        "string" => Some("string"),
        "boolean" => Some("boolean"),
        "array" => Some("array"),
        // Already-valid JSON Schema types pass through unchanged.
        "object" => Some("object"),
        "number" => Some("number"),
        // "any" and anything unrecognized → drop the constraint.
        _ => None,
    }
}

/// Sanitize a tool id into a provider-acceptable function name. Mirrors
/// `sanitizeToolName` in `agent/src/agents/_shared.ts` exactly so the de-collision
/// guarantee made here holds at the agent's registration seam.
fn sanitize_tool_name(id: &str) -> String {
    let is_safe = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    if !id.is_empty() && id.chars().all(is_safe) {
        return id.to_string();
    }
    let replaced: String = id
        .chars()
        .map(|c| if is_safe(c) { c } else { '_' })
        .collect();
    replaced.trim_matches(|c| c == '_' || c == '-').to_string()
}

/// Ensure no two *distinct* tool ids in this corpus collapse to the same
/// sanitized token (which would make the agent's `registerDirect` throw on a
/// full-universe pool). Distinct ids are processed in sorted order; the first to
/// claim a token keeps its raw id, later colliders get a deterministic `_N`
/// suffix on their `id` (the descriptive `name` is left untouched). Every
/// `candidate_pool` id and `gold_tools` entry is remapped. Returns the number of
/// ids renamed.
fn disambiguate_ids(scenarios: &mut [Scenario]) -> usize {
    let mut distinct: BTreeSet<String> = BTreeSet::new();
    for s in scenarios.iter() {
        for t in &s.candidate_pool {
            distinct.insert(t.id.clone());
        }
    }

    let mut claimed: HashSet<String> = HashSet::new();
    let mut remap: HashMap<String, String> = HashMap::new();
    for raw in &distinct {
        let token = sanitize_tool_name(raw);
        if claimed.insert(token) {
            continue; // First to claim this token keeps its raw id.
        }
        let mut k = 2usize;
        loop {
            let candidate = format!("{raw}_{k}");
            if claimed.insert(sanitize_tool_name(&candidate)) {
                remap.insert(raw.clone(), candidate);
                break;
            }
            k += 1;
        }
    }

    if remap.is_empty() {
        return 0;
    }
    for s in scenarios.iter_mut() {
        for t in s.candidate_pool.iter_mut() {
            if let Some(new_id) = remap.get(&t.id) {
                t.id = new_id.clone();
            }
        }
        for g in s.gold_tools.iter_mut() {
            if let Some(new_id) = remap.get(g) {
                *g = new_id.clone();
            }
        }
    }
    remap.len()
}

fn write_scenarios(output: &Path, scenarios: &[Scenario]) -> anyhow::Result<()> {
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating output dir {}", parent.display()))?;
    }
    let file = File::create(output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = BufWriter::new(file);
    for s in scenarios {
        writeln!(writer, "{}", serde_json::to_string(s)?)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn under_fixtures_dir_mirrors_upstream_layout() {
        let p = BfclPaths::under_fixtures_dir(Path::new("/tmp/fx"));
        assert_eq!(p.simple_data, PathBuf::from("/tmp/fx/BFCL_v3_simple.json"));
        assert_eq!(
            p.simple_answer,
            PathBuf::from("/tmp/fx/possible_answer/BFCL_v3_simple.json")
        );
        assert_eq!(p.download_targets().len(), 4);
    }

    #[test]
    fn normalize_rewrites_bfcl_types_to_json_schema() {
        let schema = json!({
            "type": "dict",
            "properties": {
                "ratio": { "type": "float", "description": "a ratio" },
                "count": { "type": "integer" },
                "tags": { "type": "array", "items": { "type": "string" } },
                "pair": { "type": "tuple" },
                "whatever": { "type": "any" }
            },
            "required": ["count"]
        });
        let n = normalize_bfcl_schema(schema);
        assert_eq!(n["type"], "object");
        assert_eq!(n["properties"]["ratio"]["type"], "number");
        assert_eq!(n["properties"]["ratio"]["description"], "a ratio");
        assert_eq!(n["properties"]["count"]["type"], "integer");
        assert_eq!(n["properties"]["tags"]["type"], "array");
        assert_eq!(n["properties"]["tags"]["items"]["type"], "string");
        assert_eq!(n["properties"]["pair"]["type"], "array");
        // "any" drops the type constraint entirely.
        assert!(n["properties"]["whatever"].get("type").is_none());
        assert_eq!(n["required"], json!(["count"]));
    }

    #[test]
    fn flatten_prompt_joins_user_messages() {
        let q = vec![vec![
            RawMessage {
                role: "system".into(),
                content: "ignored".into(),
            },
            RawMessage {
                role: "user".into(),
                content: "find the area".into(),
            },
        ]];
        assert_eq!(flatten_prompt(&q), "find the area");
    }

    fn parse_data_str(s: &str) -> Vec<RawBfclRow> {
        parse_data(s.as_bytes()).unwrap()
    }

    fn parse_answers_str(s: &str) -> HashMap<String, Vec<String>> {
        parse_answers(s.as_bytes()).unwrap()
    }

    #[test]
    fn parse_answers_extracts_gold_fn_names() {
        let answers = parse_answers_str(
            r#"{"id": "simple_0", "ground_truth": [{"calculate_triangle_area": {"base": [10]}}]}
{"id": "simple_1", "ground_truth": [{"math.factorial": {"number": [5]}}]}"#,
        );
        assert_eq!(answers["simple_0"], vec!["calculate_triangle_area"]);
        assert_eq!(answers["simple_1"], vec!["math.factorial"]);
    }

    #[test]
    fn build_scenario_maps_row_to_normalized_scenario() {
        let rows = parse_data_str(
            r#"{"id": "simple_0", "question": [[{"role": "user", "content": "area?"}]], "function": [{"name": "calc_area", "description": " desc ", "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}}}}]}"#,
        );
        let gold = vec!["calc_area".to_string()];
        let s = build_scenario(&rows[0], &gold, "bfcl-simple", "bfcl-simple").unwrap();
        assert_eq!(s.id, "bfcl-simple-simple_0");
        assert_eq!(s.prompt, "area?");
        assert_eq!(s.gold_tools, vec!["calc_area".to_string()]);
        assert_eq!(s.candidate_pool.len(), 1);
        assert_eq!(s.candidate_pool[0].id, "calc_area");
        assert_eq!(s.candidate_pool[0].description, "desc");
        // Schema normalized: dict → object.
        assert_eq!(s.candidate_pool[0].input_schema["type"], "object");
        assert_eq!(s.category.as_deref(), Some("bfcl-simple"));
    }

    #[test]
    fn build_scenario_drops_unknown_gold() {
        let rows = parse_data_str(
            r#"{"id": "x", "question": [[{"role": "user", "content": "q"}]], "function": [{"name": "a", "description": "d", "parameters": {}}]}"#,
        );
        // Gold references a function not offered in the row.
        assert!(build_scenario(&rows[0], &["b".into()], "bfcl-simple", "bfcl-simple").is_none());
    }

    #[test]
    fn multiple_keeps_all_candidate_functions() {
        let rows = parse_data_str(
            r#"{"id": "multiple_0", "question": [[{"role": "user", "content": "q"}]], "function": [{"name": "a", "description": "da", "parameters": {}}, {"name": "b", "description": "db", "parameters": {}}, {"name": "c", "description": "dc", "parameters": {}}]}"#,
        );
        let s = build_scenario(&rows[0], &["b".into()], "bfcl-multiple", "bfcl-multiple").unwrap();
        assert_eq!(s.candidate_pool.len(), 3);
        assert_eq!(s.gold_tools, vec!["b".to_string()]);
    }

    fn scenario_with_tool(id: &str, tool_id: &str, gold: &str) -> Scenario {
        Scenario {
            id: id.into(),
            prompt: "p".into(),
            candidate_pool: vec![ToolSpec {
                id: tool_id.into(),
                name: tool_id.into(),
                description: "d".into(),
                input_schema: json!({}),
                output_schema: json!({}),
            }],
            gold_tools: vec![gold.into()],
            judge_criteria: None,
            category: Some("bfcl-simple".into()),
        }
    }

    #[test]
    fn disambiguate_renames_sanitization_collisions() {
        // "math.factorial" and "math_factorial" both sanitize to "math_factorial".
        let mut scenarios = vec![
            scenario_with_tool("s1", "math.factorial", "math.factorial"),
            scenario_with_tool("s2", "math_factorial", "math_factorial"),
        ];
        let renamed = disambiguate_ids(&mut scenarios);
        assert_eq!(renamed, 1);
        // First in sorted order ("math.factorial") keeps its raw id.
        assert_eq!(scenarios[0].candidate_pool[0].id, "math.factorial");
        // The collider was renamed in BOTH its candidate_pool id and gold_tools.
        let renamed_id = &scenarios[1].candidate_pool[0].id;
        assert_ne!(renamed_id, "math_factorial");
        assert_eq!(&scenarios[1].gold_tools[0], renamed_id);
        // The descriptive name is left untouched.
        assert_eq!(scenarios[1].candidate_pool[0].name, "math_factorial");
        // Final ids sanitize to distinct tokens.
        assert_ne!(
            sanitize_tool_name(&scenarios[0].candidate_pool[0].id),
            sanitize_tool_name(&scenarios[1].candidate_pool[0].id),
        );
    }

    #[test]
    fn disambiguate_noop_when_no_collision() {
        let mut scenarios = vec![
            scenario_with_tool("s1", "alpha", "alpha"),
            scenario_with_tool("s2", "beta", "beta"),
        ];
        assert_eq!(disambiguate_ids(&mut scenarios), 0);
        assert_eq!(scenarios[0].candidate_pool[0].id, "alpha");
    }

    #[test]
    fn sanitize_matches_ts_rules() {
        assert_eq!(sanitize_tool_name("math.factorial"), "math_factorial");
        assert_eq!(sanitize_tool_name("plain_name-1"), "plain_name-1");
        assert_eq!(sanitize_tool_name("a.b.c"), "a_b_c");
    }
}
