use anthropic_bridge::config::{SystemPromptAction, SystemPromptRule, ToolFilterConfig};
use anthropic_bridge::middleware::{
    system_prompt::SystemPromptPatcherMiddleware, tool_enforcer::ToolEnforcerMiddleware,
    tool_filter::ToolFilterMiddleware, Middleware,
};
use anthropic_bridge::protocol::{
    AnthropicContentBlock, AnthropicMessageRequest, AnthropicMessageResponse, AnthropicTool,
    AnthropicToolChoice, SystemPrompt,
};
use serde_json::json;

#[test]
fn test_tool_enforcer_injects_exit_tool_and_forces_choice() {
    let mut req = AnthropicMessageRequest {
        model: "claude-3-opus-20240229".to_string(),
        messages: vec![],
        max_tokens: Some(1024),
        metadata: None,
        stop_sequences: None,
        stream: None,
        system: None,
        temperature: None,
        tool_choice: None,
        tools: Some(vec![AnthropicTool {
            name: "test_tool".to_string(),
            description: None,
            input_schema: json!({"type": "object"}),
            input_examples: None,
        }]),
        top_k: None,
        top_p: None,
        thinking: None,
    };

    let middleware = ToolEnforcerMiddleware::new();
    middleware.on_request(&mut req).unwrap();

    // Verify ExitTool injection
    let tools = req.tools.as_ref().unwrap();
    assert!(tools.iter().any(|t| t.name == "ExitTool"));

    // Verify forced tool choice
    match req.tool_choice {
        Some(AnthropicToolChoice::Any) => assert!(true),
        _ => assert!(false, "Tool choice should be forced to Any"),
    }
}

#[test]
fn test_tool_enforcer_response_interception() {
    let mut resp = AnthropicMessageResponse {
        id: "msg_123".to_string(),
        r#type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![AnthropicContentBlock::ToolUse {
            id: "call_exit".to_string(),
            name: "ExitTool".to_string(),
            input: json!({"response": "Final Answer"}),
        }],
        model: "model".to_string(),
        stop_reason: Some("tool_use".to_string()),
        stop_sequence: None,
        usage: anthropic_bridge::protocol::AnthropicUsage { input_tokens: 10, output_tokens: 10 },
    };

    let middleware = ToolEnforcerMiddleware::new();
    middleware.on_response(&mut resp).unwrap();

    assert_eq!(resp.content.len(), 1);
    match &resp.content[0] {
        AnthropicContentBlock::Text { text } => assert_eq!(text, "Final Answer"),
        _ => assert!(false, "Content should be converted to text"),
    }
    assert_eq!(resp.stop_reason, Some("end_turn".to_string()));
}

#[test]
fn test_tool_filter_middleware() {
    let mut req = AnthropicMessageRequest {
        model: "claude-3-opus-20240229".to_string(),
        messages: vec![],
        max_tokens: None,
        metadata: None,
        stop_sequences: None,
        stream: None,
        system: None,
        temperature: None,
        tool_choice: None,
        tools: Some(vec![
            AnthropicTool {
                name: "allowed_tool".to_string(),
                description: None,
                input_schema: json!({}),
                input_examples: None,
            },
            AnthropicTool {
                name: "denied_tool".to_string(),
                description: None,
                input_schema: json!({}),
                input_examples: None,
            },
            AnthropicTool {
                name: "other_tool".to_string(),
                description: None,
                input_schema: json!({}),
                input_examples: None,
            },
        ]),
        top_k: None,
        top_p: None,
        thinking: None,
    };

    let config = ToolFilterConfig {
        allow: Some(vec!["allowed_tool".to_string(), "other*".to_string()]),
        deny: Some(vec!["denied_tool".to_string()]),
    };

    let middleware = ToolFilterMiddleware::new(Some(config));
    middleware.on_request(&mut req).unwrap();

    let tools = req.tools.unwrap();
    assert!(tools.iter().any(|t| t.name == "allowed_tool"));
    assert!(tools.iter().any(|t| t.name == "other_tool"));
    assert!(!tools.iter().any(|t| t.name == "denied_tool"));
}

#[test]
fn test_system_prompt_patcher_regex_replace() {
    let mut req = AnthropicMessageRequest {
        model: "claude-3-opus-20240229".to_string(),
        messages: vec![],
        max_tokens: None,
        metadata: None,
        stop_sequences: None,
        stream: None,
        system: Some(SystemPrompt::String("You are a helpful assistant.".to_string())),
        temperature: None,
        tool_choice: None,
        tools: None,
        top_k: None,
        top_p: None,
        thinking: None,
    };

    let rules = vec![SystemPromptRule {
        name: "replace_rule".to_string(),
        r#match: vec!["helpful assistant".to_string()],
        action: SystemPromptAction::Replace("You are a grumpy cat.".to_string()),
    }];

    let middleware = SystemPromptPatcherMiddleware::new(rules);
    middleware.on_request(&mut req).unwrap();

    if let Some(SystemPrompt::String(s)) = req.system {
        assert_eq!(s, "You are a grumpy cat.");
    } else {
        assert!(false, "System prompt should be a string");
    }
}

#[test]
fn test_system_prompt_patcher_move_to_user() {
    let mut req = AnthropicMessageRequest {
        model: "claude-3-opus-20240229".to_string(),
        messages: vec![anthropic_bridge::protocol::AnthropicMessage {
            role: "user".to_string(),
            content: anthropic_bridge::protocol::AnthropicMessageContent::String("Hello".to_string()),
        }],
        max_tokens: None,
        metadata: None,
        stop_sequences: None,
        stream: None,
        system: Some(SystemPrompt::String("System Instructions".to_string())),
        temperature: None,
        tool_choice: None,
        tools: None,
        top_k: None,
        top_p: None,
        thinking: None,
    };

    let rules = vec![SystemPromptRule {
        name: "move_rule".to_string(),
        r#match: vec![".*".to_string()],
        action: SystemPromptAction::MoveToUser(Some("Simplified System".to_string())),
    }];

    let middleware = SystemPromptPatcherMiddleware::new(rules);
    middleware.on_request(&mut req).unwrap();

    // Verify system prompt replaced
    if let Some(SystemPrompt::String(s)) = req.system {
        assert_eq!(s, "Simplified System");
    }

    // Verify user message prepended
    let user_msg = &req.messages[0];
    if let anthropic_bridge::protocol::AnthropicMessageContent::String(s) = &user_msg.content {
        assert!(s.contains("System Instructions"));
        assert!(s.contains("Hello"));
    } else {
        assert!(false, "User content should be string");
    }
}
