use ratel_benchmark_retrieval::corpus::{SkillSpec, ToolSpec};
use ratel_benchmark_retrieval::retrieval::{evaluate_at_ks, evaluate_skills_at_ks};
use serde_json::json;

#[test]
fn tool_fixture_keeps_the_gold_result_first() {
    let tools = vec![
        ToolSpec {
            id: "calendar.schedule".into(),
            name: "schedule_event".into(),
            description: "Create a calendar event".into(),
            input_schema: json!({}),
            output_schema: json!({}),
        },
        ToolSpec {
            id: "weather.forecast".into(),
            name: "weather_forecast".into(),
            description: "Fetch a weather forecast".into(),
            input_schema: json!({}),
            output_schema: json!({}),
        },
    ];
    let result = evaluate_at_ks(
        &tools,
        "weather forecast",
        &["weather.forecast".into()],
        &[1],
    );
    assert_eq!(result[0].retrieved[0].id, "weather.forecast");
    assert!(result[0].hit_at_k);
}

#[test]
fn skill_fixture_keeps_the_gold_result_first() {
    let skills = vec![
        SkillSpec {
            id: "calendar".into(),
            name: "schedule event".into(),
            description: "Create a calendar event".into(),
            tags: vec![],
            tools: vec![],
            body: "Calendar instructions".into(),
        },
        SkillSpec {
            id: "weather".into(),
            name: "weather forecast".into(),
            description: "Fetch a weather forecast".into(),
            tags: vec![],
            tools: vec![],
            body: "Weather instructions".into(),
        },
    ];
    let result = evaluate_skills_at_ks(&skills, "weather forecast", &["weather".into()], &[1]);
    assert_eq!(result[0].retrieved[0].id, "weather");
    assert!(result[0].hit_at_k);
}
