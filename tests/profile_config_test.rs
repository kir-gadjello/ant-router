use anthropic_bridge::config::{Config, PreprocessConfig, Profile, Rule, merge_preprocess};
use anthropic_bridge::handlers::AppState;
use anthropic_bridge::protocol::{AnthropicMessageRequest, AnthropicTool, AnthropicToolDef};
use anthropic_bridge::transformer::request::convert_request;
use serde_json::json;
use std::collections::HashMap;

fn default_req() -> AnthropicMessageRequest {
    AnthropicMessageRequest {
        model: "claude-3-opus-20240229".to_string(),
        messages: vec![],
        max_tokens: None,
        metadata: None,
        stop_sequences: None,
        stream: None,
        system: None,
        temperature: None,
        tool_choice: None,
        tools: None,
        top_k: None,
        top_p: None,
        thinking: None,
    }
}

#[test]
fn test_profile_preprocess_override() {
    // 1. Setup Profile with explicit preprocess override
    let profile = Profile {
        rules: vec![Rule {
            pattern: ".*".to_string(),
            match_features: vec![],
            target: "default_model".to_string(),
            reasoning_target: None,
        }],
        tool_filters: None,
        system_prompts: None,
        enable_exit_tool: None,
        preprocess: Some(PreprocessConfig {
            disable_parallel_tool_calls: Some(true),
            strict_tools: Some(true),
            ..Default::default()
        }),
    };

    // 2. Setup Model Config (defaults to false/None)
    let model_config = anthropic_bridge::config::ModelConfig {
        preprocess: Some(PreprocessConfig {
            disable_parallel_tool_calls: Some(false), // Should be overridden
            ..Default::default()
        }),
        ..Default::default()
    };

    // 3. Simulate Runtime Merge (logic from handlers.rs)
    let mut merged_config = model_config.clone();
    if let Some(profile_preprocess) = &profile.preprocess {
        merged_config.preprocess = merge_preprocess(merged_config.preprocess, Some(profile_preprocess.clone()));
    }

    // 4. Verify Merge
    {
        let pp = merged_config.preprocess.as_ref().unwrap();
        assert_eq!(pp.disable_parallel_tool_calls, Some(true));
        assert_eq!(pp.strict_tools, Some(true));
    }

    // 5. Verify Request Conversion uses merged config
    let mut req = default_req();
    req.tools = Some(vec![AnthropicTool::Anthropic(AnthropicToolDef {
        name: "Edit".to_string(),
        description: None,
        input_schema: json!({"type": "object", "properties": {}}),
        input_examples: None,
        strict: None,
    })]);

    let (openai_req, _) = convert_request(req, "test_model".to_string(), Some(&merged_config)).unwrap();

    // Check parallel tool calls disabled
    assert_eq!(openai_req.parallel_tool_calls, Some(false));

    // Check Edit tool is strict
    let tools = openai_req.tools.unwrap();
    let edit_tool = tools.iter().find(|t| t.function.name == "Edit").unwrap();
    assert_eq!(edit_tool.function.strict, Some(true));
}
