use anthropic_bridge::protocol::*;
use anthropic_bridge::transformer::request::{convert_request};
use anthropic_bridge::config::{ModelConfig, PreprocessConfig};
use serde_json::{json, Value};

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
fn test_sanitize_tool_history_orphaned_result_from_missing_tool_use() {
    let mut req = default_req();
    req.messages = vec![
        AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::String("Hello".to_string()),
        },
        AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::Blocks(vec![
                AnthropicContentBlock::ToolResult {
                    tool_use_id: "non_existent_id".to_string(),
                    content: Some(AnthropicMessageContent::String("Result".to_string())),
                    is_error: None,
                }
            ]),
        },
    ];

    let preprocess_config = PreprocessConfig {
        merge_system_messages: None,
        sanitize_tool_history: Some(true),
        max_output_tokens: None,
        max_output_cap: None,
    };

    let model_config = ModelConfig {
        preprocess: Some(preprocess_config),
        capabilities: None,
        force_reasoning: None,
        min_reasoning: None,
        max_tokens: None,
        override_max_tokens: None,
        ..Default::default()
    };

    let (openai_req, _) = convert_request(req, "test_model".to_string(), Some(&model_config)).unwrap();

    // Expectation: This should be removed by sanitize_tool_history
    let tool_msg = openai_req.messages.iter().find(|m| m.role == "tool" && m.tool_call_id == Some("non_existent_id".to_string()));

    assert!(tool_msg.is_none(), "Orphaned tool result should be removed");
}

#[test]
fn test_tool_result_with_image_content() {
    let mut req = default_req();
    req.messages = vec![
        AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::Blocks(vec![
                AnthropicContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::Text { text: "Here is the image".to_string() },
                        AnthropicContentBlock::Image {
                            source: AnthropicImageSource::Base64 {
                                media_type: "image/png".to_string(),
                                data: "base64data".to_string()
                            }
                        }
                    ])),
                    is_error: None,
                }
            ]),
        },
    ];

    // No sanitation needed for this test
    let (openai_req, _) = convert_request(req, "test_model".to_string(), None).unwrap();

    let tool_msg = openai_req.messages.iter().find(|m| m.role == "tool" && m.tool_call_id == Some("call_1".to_string())).unwrap();

    if let Some(OpenAIContent::String(s)) = &tool_msg.content {
        // New behavior: should contain JSON string of blocks
        assert!(s.contains("Here is the image"), "Should contain text");
        assert!(s.contains("image/png"), "Should contain image media type");
        assert!(s.contains("base64data"), "Should contain base64 data");
        // Ensure it's valid JSON
        serde_json::from_str::<Value>(s).expect("Content should be valid JSON string");
    } else {
        panic!("Expected String content for tool message");
    }
}

#[test]
fn test_sanitize_tool_history_removes_empty_name() {
    let mut req = default_req();
    req.messages = vec![
        AnthropicMessage {
            role: "assistant".to_string(),
            content: AnthropicMessageContent::Blocks(vec![
                AnthropicContentBlock::ToolUse {
                    id: "bad_tool".to_string(),
                    name: "".to_string(), // Empty name
                    input: json!({}),
                }
            ]),
        },
        AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::Blocks(vec![
                AnthropicContentBlock::ToolResult {
                    tool_use_id: "bad_tool".to_string(),
                    content: Some(AnthropicMessageContent::String("Result".to_string())),
                    is_error: None,
                }
            ]),
        },
    ];

    let preprocess_config = PreprocessConfig {
        merge_system_messages: None,
        sanitize_tool_history: Some(true),
        max_output_tokens: None,
        max_output_cap: None,
    };

    let model_config = ModelConfig {
        preprocess: Some(preprocess_config),
        capabilities: None,
        force_reasoning: None,
        min_reasoning: None,
        max_tokens: None,
        override_max_tokens: None,
        ..Default::default()
    };

    let (openai_req, report) = convert_request(req, "test_model".to_string(), Some(&model_config)).unwrap();

    assert!(report.sanitized_tool_ids.contains(&"bad_tool".to_string()));

    // Check that tool use is gone from assistant message
    let assistant_msg = openai_req.messages.iter().find(|m| m.role == "assistant");
    if let Some(msg) = assistant_msg {
        if let Some(tool_calls) = &msg.tool_calls {
            assert!(tool_calls.iter().all(|tc| tc.id != "bad_tool"));
        }
    }

    // Check that tool result is gone
    let tool_result_msg = openai_req.messages.iter().find(|m| m.role == "tool" && m.tool_call_id == Some("bad_tool".to_string()));
    assert!(tool_result_msg.is_none(), "Tool result should be removed because corresponding tool use was sanitized");
}

#[test]
fn test_tool_strict_mapping() {
    let mut req = default_req();
    req.tools = Some(vec![
        AnthropicTool {
            name: "strict_tool".to_string(),
            description: Some("A strict tool".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": { "foo": { "type": "string" } }
            }),
            input_examples: None,
            strict: Some(true),
        }
    ]);

    let (openai_req, _) = convert_request(req, "test_model".to_string(), None).unwrap();

    let tools = openai_req.tools.unwrap();
    let tool = &tools[0];
    assert_eq!(tool.function.strict, Some(true));
}
