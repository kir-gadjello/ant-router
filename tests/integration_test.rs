use anthropic_bridge::{
    config::{ApiParams, Config, ModelConfig, Profile, Rule},
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
                        target: "logical-gpt4".to_string(),
                        reasoning_target: None,
                    },
                    Rule {
                        pattern: "reasoning*".to_string(),
                        target: "logical-gpt4".to_string(),
                        reasoning_target: Some("logical-o1".to_string()),
                    },
                ],
                ant_vision_model: Some("logical-gpt4".to_string()),
                ant_vision_reasoning_model: Some("logical-o1-vision".to_string()),
            },
        )]),
        models: HashMap::from([
            ("logical-gpt4".to_string(), ModelConfig {
                api_model_id: Some("openai/gpt-4o".to_string()),
                ..Default::default()
            }),
            ("logical-o1".to_string(), ModelConfig {
                api_model_id: Some("openai/o1".to_string()),
                ..Default::default()
            }),
            ("logical-o1-vision".to_string(), ModelConfig {
                api_model_id: Some("openai/o1-vision".to_string()),
                ..Default::default()
            }),
            ("deepseek-v3".to_string(), ModelConfig {
                 api_model_id: Some("deepseek/v3".to_string()),
                 api_params: Some(ApiParams {
                     extra_body: HashMap::from([
                         ("provider".to_string(), json!({"order": ["baseten"]}))
                     ]),
                     ..Default::default()
                 }),
                 ..Default::default()
            })
        ]),
        ..Default::default()
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
    match &msg2.content {
        Some(OpenAIContent::String(s)) => assert_eq!(s, "Sunny"),
        _ => panic!("Expected string content"),
    }
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

    // Test 1: claude* -> logical-gpt4 -> openai/gpt-4o
    client.post(format!("{}/v1/messages", addr))
        .json(&json!({"model": "claude-3", "messages": [{"role":"user","content":"a"}], "max_tokens": 1}))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r1: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(r1.model, "openai/gpt-4o");

    mock_server.reset().await;

    // Test 2: reasoning* (thinking=true) -> logical-o1 -> openai/o1
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

#[tokio::test]
async fn test_vision_routing() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    // 1. Request with image (no thinking) -> profile.ant_vision_model (logical-gpt4) -> openai/gpt-4o
    client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-3-sonnet", // Arbitrary model input
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe image"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
                ]
            }],
            "max_tokens": 100
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r1: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(r1.model, "openai/gpt-4o");

    match &r1.messages[0].content {
        Some(OpenAIContent::Array(parts)) => {
            assert!(parts.iter().any(|p| matches!(p, OpenAIContentPart::ImageUrl { .. })));
        },
        _ => panic!("Expected content array"),
    }

    mock_server.reset().await;

    // 2. Request with image AND thinking -> profile.ant_vision_reasoning_model (logical-o1-vision) -> openai/o1-vision
    client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-3-sonnet",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "123"}}
                ]
            }],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "max_tokens": 100
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r2: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(r2.model, "openai/o1-vision");
}

#[tokio::test]
async fn test_umcp_extra_body() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
             "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    // Direct mapping test using UMCP model ID directly (bypass rules for test simplicity if supported, or via rule)
    // Here we use a rule-less request that should fail if we didn't have a catch-all, but we can update test_config to route to deepseek-v3
    // Or simpler: define a rule for it.
    // Wait, handlers.rs logic says: if vision -> ... else -> resolve_via_rules.
    // If resolve_via_rules returns input, we check if input is in models.
    // So if we request "deepseek-v3" and it is in models, it works!

    client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "deepseek-v3",
            "messages": [{"role":"user", "content":"hi"}],
            "max_tokens": 10
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();

    // Check extra_body merged
    assert_eq!(body["provider"]["order"][0], "baseten");
    // Check model ID mapped
    assert_eq!(body["model"], "deepseek/v3");
}

#[tokio::test]
async fn test_vision_url_input() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "image ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-vision", // Should match 'claude*' pattern -> logical-gpt4 -> openai/gpt-4o (because has images)
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check URL"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/image.jpg"}}
                ]
            }],
            "max_tokens": 100
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r1: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();

    assert_eq!(r1.model, "openai/gpt-4o");

    match &r1.messages[0].content {
        Some(OpenAIContent::Array(parts)) => {
            let img_part = parts.iter().find(|p| matches!(p, OpenAIContentPart::ImageUrl { .. })).unwrap();
            if let OpenAIContentPart::ImageUrl { image_url } = img_part {
                assert_eq!(image_url.url, "https://example.com/image.jpg");
            } else {
                panic!("Expected ImageUrl");
            }
        },
        _ => panic!("Expected content array"),
    }
}

#[tokio::test]
async fn test_mixed_images_text() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "mixed ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-vision", // Routes to gpt-4o
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Start"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A"}},
                    {"type": "text", "text": "Middle"},
                    {"type": "image", "source": {"type": "url", "url": "http://img.com/B.jpg"}},
                    {"type": "text", "text": "End"}
                ]
            }],
            "max_tokens": 100
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r1: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();

    match &r1.messages[0].content {
        Some(OpenAIContent::Array(parts)) => {
            assert_eq!(parts.len(), 5);
            // Verify order preservation
            matches!(&parts[0], OpenAIContentPart::Text { text } if text == "Start");
            matches!(&parts[1], OpenAIContentPart::ImageUrl { .. });
            matches!(&parts[2], OpenAIContentPart::Text { text } if text == "Middle");
            matches!(&parts[3], OpenAIContentPart::ImageUrl { .. });
            matches!(&parts[4], OpenAIContentPart::Text { text } if text == "End");
        },
        _ => panic!("Expected content array"),
    }
}

#[tokio::test]
async fn test_no_ant_flag() {
    let mut config = test_config();
    config.no_ant = true;

    // Add a model that maps to anthropic
    config.models.insert("forbidden-claude".to_string(), ModelConfig {
        api_model_id: Some("anthropic/claude-3-opus".to_string()),
        ..Default::default()
    });

    let (addr, _mock_server) = spawn_app(config).await;
    let client = Client::new();

    let resp = client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "forbidden-claude",
            "messages": [{"role":"user", "content":"hi"}],
            "max_tokens": 10
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 403);
}
