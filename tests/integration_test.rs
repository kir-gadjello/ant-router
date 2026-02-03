use anthropic_bridge::{
    config::{Config, Profile, Rule},
    create_router,
    handlers::AppState,
    protocol::*,
};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// Helper to start the server
async fn spawn_app(config: Config) -> (String, MockServer) {
    let mock_server = MockServer::start().await;

    let app_state = Arc::new(AppState {
        config,
        client: Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("test-key".to_string()),
    });

    let app = create_router(app_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{}", addr), mock_server)
}

fn test_config() -> Config {
    Config {
        current_profile: "test".to_string(),
        profiles: HashMap::from([(
            "test".to_string(),
            Profile {
                rules: vec![
                    Rule {
                        pattern: "claude*".to_string(),
                        target: "openai/gpt-4o".to_string(),
                        reasoning_target: None,
                    },
                    Rule {
                        pattern: "reasoning*".to_string(),
                        target: "openai/gpt-4o".to_string(),
                        reasoning_target: Some("openai/o1".to_string()),
                    },
                ],
            },
        )]),
    }
}

#[tokio::test]
async fn test_basic_message_echo() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    // Mock upstream response
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "openai/gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })))
        .mount(&mock_server)
        .await;

    // Send request to bridge
    let resp = client
        .post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-3-5-sonnet",
            "messages": [{
                "role": "user",
                "content": "Hi"
            }],
            "max_tokens": 1024
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: AnthropicMessageResponse = resp.json().await.unwrap();

    assert_eq!(body.role, "assistant");
    assert_eq!(body.id, "msg-123"); // chatcmpl replaced by msg

    // Check content
    match &body.content[0] {
        AnthropicContentBlock::Text { text } => assert_eq!(text, "Hello world"),
        _ => panic!("Expected text block"),
    }
}

#[tokio::test]
async fn test_tool_call_round_trip() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    // We expect the bridge to receive a request with tool results and tool use
    // And forward it to upstream as tool messages.

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Result received"
                },
                "finish_reason": "stop"
            }]
        })))
        .mount(&mock_server)
        .await;

    let resp = client
        .post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-3",
            "messages": [
                {
                    "role": "user",
                    "content": "use tool"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "get_weather",
                            "input": {"city": "London"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "Sunny"
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());

    // Verify what upstream received
    let requests = mock_server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);

    let upstream_req: OpenAIChatCompletionRequest =
        serde_json::from_slice(&requests[0].body).unwrap();

    // Check messages order
    // 0: user "use tool"
    // 1: assistant tool_calls
    // 2: tool message (from tool_result)

    assert_eq!(upstream_req.messages.len(), 3);

    let msg1 = &upstream_req.messages[1];
    assert_eq!(msg1.role, "assistant");
    assert!(msg1.tool_calls.is_some());

    let msg2 = &upstream_req.messages[2];
    assert_eq!(msg2.role, "tool");
    assert_eq!(msg2.tool_call_id.as_deref(), Some("tool_1"));
    assert_eq!(msg2.content.as_deref(), Some("Sunny"));
}

#[tokio::test]
async fn test_streaming() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    // Mock SSE response
    // data: {...}
    // data: {...}
    // data: [DONE]
    let sse_body = "data: {\"id\":\"chatcmpl-s\",\"object\":\"chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n\
                    data: {\"id\":\"chatcmpl-s\",\"object\":\"chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n\
                    data: {\"id\":\"chatcmpl-s\",\"object\":\"chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" World\"}}]}\n\n\
                    data: [DONE]\n\n";

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&mock_server)
        .await;

    let resp = client
        .post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-3",
            "messages": [{"role":"user", "content":"hi"}],
            "stream": true,
            "max_tokens": 100
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body = resp.text().await.unwrap();

    // Verify SSE events
    assert!(body.contains("event: message_start"));
    assert!(body.contains("event: content_block_start"));
    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("event: message_stop"));

    // Simple check if Hello and World are there
    assert!(body.contains("Hello"));
    assert!(body.contains(" World"));
}

#[tokio::test]
async fn test_config_resolution() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    // Test 1: claude* -> openai/gpt-4o
    client.post(format!("{}/v1/messages", addr))
        .json(&json!({"model": "claude-3", "messages": [{"role":"user","content":"a"}], "max_tokens": 1}))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r1: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(r1.model, "openai/gpt-4o");

    mock_server.reset().await;

    // Test 2: reasoning* (thinking=true) -> openai/o1
    client
        .post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "reasoning-claude",
            "messages": [{"role":"user","content":"a"}],
            "max_tokens": 1,
            "thinking": {"type": "enabled", "budget_tokens": 100}
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r2: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(r2.model, "openai/o1");
}
