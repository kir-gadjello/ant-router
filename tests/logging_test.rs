use anthropic_bridge::{
    config::Config,
    logging::log_request,
    protocol::{AnthropicMessage, AnthropicMessageContent, AnthropicMessageRequest},
};
use std::fs;
use tempfile::NamedTempFile;

#[test]
fn test_logging_output() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let req = AnthropicMessageRequest {
        model: "claude-3-opus".to_string(),
        messages: vec![AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::String("test log".to_string()),
        }],
        max_tokens: Some(10),
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
    };

    log_request(&req, path).unwrap();

    let content = fs::read_to_string(path).unwrap();
    assert!(content.contains("_metadata"));
    assert!(content.contains("claude-3-opus"));
    assert!(content.contains("test log"));

    // Verify JSON validity
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("_metadata").is_some());
    assert!(json.get("_metadata").unwrap().get("timestamp").is_some());
}

#[tokio::test]
async fn test_config_log_defaults() {
    // Ensure default is enabled
    let config = Config::default();
    assert!(config.log_enabled);

    // Check default path structure (relative check)
    let log_path = config.get_log_path();
    assert!(log_path
        .to_string_lossy()
        .contains(".ant-router/logs/.log.jsonl"));
}
