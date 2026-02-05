use anthropic_bridge::config::Config;
use anthropic_bridge::create_router;
use anthropic_bridge::handlers::AppState;
use anthropic_bridge::protocol::{AnthropicMessageResponse, AnthropicContentBlock};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::fs;
use serde_json::json;

#[tokio::test]
async fn test_e2e_chat_compat() {
    // 1. Setup Wiremock
    let mock_server = MockServer::start().await;

    // 2. Load Config
    let mut config = Config::load("tests/config_test_e2e.yaml").await.expect("Failed to load config");
    config.upstream.base_url = Some(mock_server.uri()); // Point to mock
    
    // 3. Setup Router
    let client = reqwest::Client::builder().build().unwrap();
    let state = Arc::new(AppState {
        config,
        client,
        base_url: mock_server.uri(),
        api_key: Some("test-key".to_string()),
        verbose: true,
        tool_verbose: false,
        debug_tools: false,
        record: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    // 4. Prepare Mock Response (Ground Truth)
    let ground_truth_body = fs::read_to_string("ground_truth_chat.json").expect("Missing ground_truth_chat.json");
    // Ensure it's valid JSON
    let _: serde_json::Value = serde_json::from_str(&ground_truth_body).expect("Invalid JSON in ground truth");

    // 5. Setup Mock Expectation
    // We expect a request to /v1/chat/completions with specific body
    // Note: ant-router might add default fields. We'll match subset.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer test-key"))
        .and(wiremock::matchers::body_partial_json(json!({
            "model": "stepfun/step-3.5-flash:free",
            "messages": [
                {"role": "user", "content": "Write a hello world in python"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(ground_truth_body))
        .mount(&mock_server)
        .await;

    // 6. Send Anthropic Request
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(json!({
                    "model": "stepfun-flash",
                    "messages": [
                        {"role": "user", "content": "Write a hello world in python"}
                    ],
                    "max_tokens": 1024
                }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // 7. Verify Response Body Conversion
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let resp_body: AnthropicMessageResponse = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(resp_body.role, "assistant");
    assert!(!resp_body.content.is_empty());

    // Verify Thinking Block (Added due to reasoning support)
    let thinking = resp_body.content.iter().find(|b| matches!(b, AnthropicContentBlock::Thinking { .. }));
    assert!(thinking.is_some(), "Should contain Thinking block");

    // Verify Text Block
    let text_block = resp_body.content.iter().find(|b| matches!(b, AnthropicContentBlock::Text { .. }));
    assert!(text_block.is_some(), "Should contain Text block");

    if let AnthropicContentBlock::Text { text } = text_block.unwrap() {
        assert!(text.contains("print(\"Hello, world!\")"));
    }
}

#[tokio::test]
async fn test_e2e_tool_compat() {
    let mock_server = MockServer::start().await;
    let mut config = Config::load("tests/config_test_e2e.yaml").await.unwrap();
    config.upstream.base_url = Some(mock_server.uri());
    
    // 3. Setup Router
    let client = reqwest::Client::builder().build().unwrap();
    let state = Arc::new(AppState {
        config,
        client,
        base_url: mock_server.uri(),
        api_key: Some("test-key".to_string()),
        verbose: true,
        tool_verbose: false,
        debug_tools: false,
        record: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    let ground_truth_body = fs::read_to_string("ground_truth_tool_call.json").expect("Missing ground_truth_tool_call.json");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_partial_json(json!({
            "model": "stepfun/step-3.5-flash:free",
            "messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather"
                    }
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(ground_truth_body))
        .mount(&mock_server)
        .await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(json!({
                    "model": "stepfun-flash",
                    "messages": [
                        {"role": "user", "content": "What is the weather in Tokyo?"}
                    ],
                    "max_tokens": 1024,
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get current weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"}
                                },
                                "required": ["location"]
                            }
                        }
                    ]
                }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let resp_body: AnthropicMessageResponse = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(resp_body.stop_reason, Some("tool_use".to_string()));
    
    // Check for Thinking block
    let thinking = resp_body.content.iter().find(|b| matches!(b, AnthropicContentBlock::Thinking { .. }));
    assert!(thinking.is_some(), "Should contain Thinking block");

    // Find tool use block
    let tool_use = resp_body.content.iter().find(|b| matches!(b, AnthropicContentBlock::ToolUse { .. }));
    assert!(tool_use.is_some(), "Should contain ToolUse block");
    
    if let AnthropicContentBlock::ToolUse { name, input, .. } = tool_use.unwrap() {
        assert_eq!(name, "get_weather");
        assert_eq!(input["location"], "Tokyo");
    }
}

#[tokio::test]
async fn test_min_reasoning_injection() {
    let mock_server = MockServer::start().await;
    let mut config = Config::load("tests/config_test_e2e.yaml").await.unwrap();
    config.upstream.base_url = Some(mock_server.uri());
    
    let client = reqwest::Client::builder().build().unwrap();
    let state = Arc::new(AppState {
        config,
        client,
        base_url: mock_server.uri(),
        api_key: Some("test-key".to_string()),
        verbose: true,
        tool_verbose: false,
        debug_tools: false,
        record: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    // Expect reasoning to be injected because profile 'stepfun-reasoning' has min_reasoning: true
    Mock::given(method("POST"))
        .and(wiremock::matchers::body_partial_json(json!({
            "model": "stepfun/step-3.5-flash:free",
            "reasoning": {
                "effort": "low" // true maps to low in our logic? "min implies at least something" -> "low"
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1",
            "object": "chat.completion",
            "created": 123,
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok"
                },
                "finish_reason": "stop"
            }]
        })))
        .mount(&mock_server)
        .await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(json!({
                    "model": "stepfun-reasoning", // Targets the model with min_reasoning
                    "messages": [
                        {"role": "user", "content": "Think please"}
                    ],
                    "max_tokens": 1024
                    // No "thinking" param sent!
                }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
