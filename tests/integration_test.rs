use anthropic_bridge::{
    config::{ApiParams, Config, ModelConfig},
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
        ant_vision_model: Some("claude-vision".to_string()),
        ant_vision_reasoning_model: Some("claude-vision-think".to_string()),
        models: HashMap::from([
            (
                "claude-vision".to_string(),
                ModelConfig {
                    api_model_id: Some("openai/gpt-4o".to_string()),
                    ..Default::default()
                },
            ),
            (
                "claude-vision-think".to_string(),
                ModelConfig {
                    api_model_id: Some("openai/o1-vision".to_string()),
                    ..Default::default()
                },
            ),
            (
                "deepseek-v3".to_string(),
                ModelConfig {
                    api_model_id: Some("deepseek/v3".to_string()),
                    api_params: Some(ApiParams {
                        extra_body: HashMap::from([(
                            "provider".to_string(),
                            json!({"order": ["baseten"]}),
                        )]),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
        ]),
        ..Default::default()
    }
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

    // 1. Request with image (no thinking) -> Should route to ant_vision_model -> openai/gpt-4o
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

    // Check if image content part is present
    match &r1.messages[0].content {
        Some(OpenAIContent::Array(parts)) => {
            assert!(parts
                .iter()
                .any(|p| matches!(p, OpenAIContentPart::ImageUrl { .. })));
        }
        _ => panic!("Expected content array"),
    }

    mock_server.reset().await;

    // 2. Request with image AND thinking -> Should route to ant_vision_reasoning_model -> openai/o1-vision
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

    // Request for deepseek-v3 which has extra_body
    client
        .post(format!("{}/v1/messages", addr))
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
            "model": "claude-vision",
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

    match &r1.messages[0].content {
        Some(OpenAIContent::Array(parts)) => {
            let img_part = parts
                .iter()
                .find(|p| matches!(p, OpenAIContentPart::ImageUrl { .. }))
                .unwrap();
            if let OpenAIContentPart::ImageUrl { image_url } = img_part {
                assert_eq!(image_url.url, "https://example.com/image.jpg");
            } else {
                panic!("Expected ImageUrl");
            }
        }
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
            "model": "claude-vision",
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
        }
        _ => panic!("Expected content array"),
    }
}
