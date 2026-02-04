use anthropic_bridge::{
    config::{ApiParams, Config, ModelConfig, Profile, Rule},
    create_router,
    handlers::AppState,
    protocol::*,
};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
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
        verbose: false,
        tool_verbose: false,
        record: false,
        tools_reported: AtomicBool::new(false),
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
                    // Feature match rule comes first
                    Rule {
                        pattern: "claude*".to_string(),
                        match_features: vec!["vision".to_string()],
                        target: "logical-gpt4-vision".to_string(),
                        reasoning_target: None,
                    },
                    // Standard match
                    Rule {
                        pattern: "claude*".to_string(),
                        match_features: vec![],
                        target: "logical-gpt4".to_string(),
                        reasoning_target: None,
                    },
                    // Reasoning match
                    Rule {
                        pattern: "reasoning*".to_string(),
                        match_features: vec!["reasoning".to_string()],
                        target: "logical-o1".to_string(),
                        reasoning_target: None,
                    },
                ],
            },
        )]),
        models: HashMap::from([
            ("logical-gpt4".to_string(), ModelConfig {
                api_model_id: Some("openai/gpt-4o".to_string()),
                ..Default::default()
            }),
            ("logical-gpt4-vision".to_string(), ModelConfig {
                api_model_id: Some("openai/gpt-4o-v".to_string()),
                ..Default::default()
            }),
            ("logical-o1".to_string(), ModelConfig {
                api_model_id: Some("openai/o1".to_string()),
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

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}}],
        })))
        .mount(&mock_server)
        .await;

    let resp = client
        .post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024
        }))
        .send()
        .await
        .unwrap();

    if !resp.status().is_success() {
        println!("Response: {}", resp.text().await.unwrap());
        panic!("Request failed");
    }

    assert!(resp.status().is_success());
    let body: AnthropicMessageResponse = resp.json().await.unwrap();
    assert_eq!(body.role, "assistant");
}

#[tokio::test]
async fn test_config_resolution() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1",
            "object": "chat.completion",
            "created": 123,
            "model": "test-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    // Test 1: claude* (no vision) -> logical-gpt4 -> openai/gpt-4o
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
            "id": "1",
            "object": "chat.completion",
            "created": 123,
            "model": "test-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    // Request with image -> matches rule with match_features: [vision] -> logical-gpt4-vision -> openai/gpt-4o-v
    client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "claude-3-sonnet",
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
    assert_eq!(r1.model, "openai/gpt-4o-v");
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

    // Add a catch-all profile rule for this test model
    let profile = config.profiles.get_mut("test").unwrap();
    profile.rules.push(Rule {
        pattern: "forbidden-claude".to_string(),
        target: "forbidden-claude".to_string(),
        match_features: vec![],
        reasoning_target: None,
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

// Reproduce bug: Default config .* matching
#[tokio::test]
async fn test_bug_default_catchall() {
    // Config imitating the default one
    let config = Config {
        current_profile: "default".to_string(),
        profiles: HashMap::from([
            ("default".to_string(), Profile {
                rules: vec![
                    Rule {
                        pattern: ".*".to_string(), // The problematic pattern
                        match_features: vec![],
                        target: "catchall".to_string(),
                        reasoning_target: None,
                    }
                ],
            })
        ]),
        models: HashMap::from([
            ("catchall".to_string(), ModelConfig {
                api_model_id: Some("caught".to_string()),
                ..Default::default()
            })
        ]),
        ..Default::default()
    };

    let (addr, mock_server) = spawn_app(config).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1",
            "object": "chat.completion",
            "created": 123,
            "model": "test-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    client.post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "anthropic/claude-sonnet-4.5", // Should match .*
            "messages": [{"role":"user","content":"a"}],
            "max_tokens": 1
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r1: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(r1.model, "caught");
}

#[tokio::test]
async fn test_reasoning_parameter_mapping() {
    let (addr, mock_server) = spawn_app(test_config()).await;
    let client = Client::new();

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    // Send Anthropic request with thinking enabled
    client
        .post(format!("{}/v1/messages", addr))
        .json(&json!({
            "model": "reasoning-claude",
            "messages": [{"role":"user","content":"Think about it"}],
            "max_tokens": 2048,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024
            }
        }))
        .send()
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let upstream_body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();

    // Verify reasoning parameter is present and correct
    let reasoning = upstream_body.get("reasoning").expect("reasoning field missing");
    assert_eq!(reasoning["max_tokens"], 1024);
}
