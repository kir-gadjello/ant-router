use anthropic_bridge::config::{Config, ModelConfig, PreprocessConfig};
use anthropic_bridge::create_router;
use anthropic_bridge::handlers::AppState;
use anthropic_bridge::protocol::{OpenAIChatCompletionRequest};
use axum::{
    body::Body,
    http::Request,
};
use tower::util::ServiceExt;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::json;

#[tokio::test]
async fn test_preprocess_merge_sysmsgs() {
    let mock_server = MockServer::start().await;
    
    let mut config = Config::default();
    config.models.insert("merged-sys".to_string(), ModelConfig {
        api_model_id: Some("test".to_string()),
        preprocess: Some(PreprocessConfig {
            merge_system_messages: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    });
    config.profiles.insert("test".to_string(), anthropic_bridge::config::Profile {
        rules: vec![anthropic_bridge::config::Rule {
            pattern: "test".to_string(),
            target: "merged-sys".to_string(),
            match_features: vec![],
            reasoning_target: None,
        }],
    });
    config.current_profile = "test".to_string();
    config.upstream.base_url = Some(mock_server.uri());

    let state = Arc::new(AppState {
        config,
        client: reqwest::Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("k".to_string()),
        verbose: true,
        tool_verbose: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    app.oneshot(Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(json!({
            "model": "test",
            "system": [
                {"type": "text", "text": "System 1"},
                {"type": "text", "text": "System 2"}
            ],
            "messages": [{"role": "user", "content": "hi"}]
        }).to_string()))
        .unwrap())
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    
    // Check if system message is merged
    // OpenAI format: system messages are in `messages`.
    let sys_msg = r.messages.iter().find(|m| m.role == "system").unwrap();
    if let Some(anthropic_bridge::protocol::OpenAIContent::String(s)) = &sys_msg.content {
        assert!(s.contains("System 1\n\nSystem 2"));
    } else {
        panic!("System message content not string");
    }
}

#[tokio::test]
async fn test_preprocess_sanitize_tool_history() {
    let mock_server = MockServer::start().await;
    
    let mut config = Config::default();
    config.models.insert("sanitize".to_string(), ModelConfig {
        api_model_id: Some("test".to_string()),
        preprocess: Some(PreprocessConfig {
            sanitize_tool_history: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    });
    config.profiles.insert("test".to_string(), anthropic_bridge::config::Profile {
        rules: vec![anthropic_bridge::config::Rule {
            pattern: "test".to_string(),
            target: "sanitize".to_string(),
            match_features: vec![],
            reasoning_target: None,
        }],
    });
    config.current_profile = "test".to_string();
    config.upstream.base_url = Some(mock_server.uri());

    let state = Arc::new(AppState {
        config,
        client: reqwest::Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("k".to_string()),
        verbose: true,
        tool_verbose: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    // Send request with invalid tool calls (empty name) and their outputs
    let body = json!({
        "model": "test",
        "messages": [
            {
                "role": "user", "content": "do bad tool"
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I will do bad tool"
                    },
                    {
                        "type": "tool_use",
                        "id": "bad_tool_1",
                        "name": "", // EMPTY NAME - Should be removed
                        "input": {}
                    },
                    {
                        "type": "tool_use",
                        "id": "good_tool_1",
                        "name": "good_tool",
                        "input": {}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "bad_tool_1",
                        "content": "Error"
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "good_tool_1",
                        "content": "Success"
                    }
                ]
            }
        ],
        "max_tokens": 100
    });

    app.oneshot(Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap())
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();

    // Verify bad_tool_1 is gone from assistant message
    let asst_msg = r.messages.iter().find(|m| m.role == "assistant").unwrap();
    let tools = asst_msg.tool_calls.as_ref().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "good_tool");

    // Verify bad_tool_1 result is gone from tool message
    // In OpenAI, tool results are separate messages with role "tool"
    let tool_msgs: Vec<_> = r.messages.iter().filter(|m| m.role == "tool").collect();
    assert_eq!(tool_msgs.len(), 1);
    assert_eq!(tool_msgs[0].tool_call_id.as_deref(), Some("good_tool_1"));
}

#[tokio::test]
async fn test_max_output_cap() {
    let mock_server = MockServer::start().await;
    
    let mut config = Config::default();
    config.models.insert("capped".to_string(), ModelConfig {
        api_model_id: Some("test".to_string()),
        preprocess: Some(PreprocessConfig {
            max_output_cap: Some(100),
            ..Default::default()
        }),
        ..Default::default()
    });
    config.profiles.insert("test".to_string(), anthropic_bridge::config::Profile {
        rules: vec![anthropic_bridge::config::Rule {
            pattern: "test".to_string(),
            target: "capped".to_string(),
            match_features: vec![],
            reasoning_target: None,
        }],
    });
    config.current_profile = "test".to_string();
    config.upstream.base_url = Some(mock_server.uri());

    let state = Arc::new(AppState {
        config,
        client: reqwest::Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("k".to_string()),
        verbose: true,
        tool_verbose: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    app.oneshot(Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5000 // Exceeds cap
        }).to_string()))
        .unwrap())
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    
    assert_eq!(r.max_tokens, Some(100));
}

#[tokio::test]
async fn test_max_tokens_override() {
    let mock_server = MockServer::start().await;
    
    let mut config = Config::default();
    config.models.insert("overridden".to_string(), ModelConfig {
        api_model_id: Some("test".to_string()),
        max_tokens: Some(50),
        ..Default::default()
    });
    config.profiles.insert("test".to_string(), anthropic_bridge::config::Profile {
        rules: vec![anthropic_bridge::config::Rule {
            pattern: "test".to_string(),
            target: "overridden".to_string(),
            match_features: vec![],
            reasoning_target: None,
        }],
    });
    config.current_profile = "test".to_string();
    config.upstream.base_url = Some(mock_server.uri());

    let state = Arc::new(AppState {
        config,
        client: reqwest::Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("k".to_string()),
        verbose: true,
        tool_verbose: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    app.oneshot(Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1000
        }).to_string()))
        .unwrap())
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    
    assert_eq!(r.max_tokens, Some(50));
}

#[tokio::test]
async fn test_max_output_tokens_auto() {
    let mock_server = MockServer::start().await;
    
    let mut config = Config::default();
    config.models.insert("auto_tokens".to_string(), ModelConfig {
        api_model_id: Some("test".to_string()),
        preprocess: Some(PreprocessConfig {
            max_output_tokens: Some(json!("auto")),
            ..Default::default()
        }),
        ..Default::default()
    });
    config.profiles.insert("test".to_string(), anthropic_bridge::config::Profile {
        rules: vec![anthropic_bridge::config::Rule {
            pattern: "test".to_string(),
            target: "auto_tokens".to_string(),
            match_features: vec![],
            reasoning_target: None,
        }],
    });
    config.current_profile = "test".to_string();
    config.upstream.base_url = Some(mock_server.uri());

    let state = Arc::new(AppState {
        config,
        client: reqwest::Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("k".to_string()),
        verbose: true,
        tool_verbose: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    app.oneshot(Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1000
        }).to_string()))
        .unwrap())
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    
    assert_eq!(r.max_tokens, None);
}

#[tokio::test]
async fn test_override_max_tokens_auto() {
    let mock_server = MockServer::start().await;
    
    let mut config = Config::default();
    config.models.insert("override_auto".to_string(), ModelConfig {
        api_model_id: Some("test".to_string()),
        override_max_tokens: Some(json!("auto")),
        capabilities: Some(anthropic_bridge::config::Capabilities {
            max_output_tokens: Some(json!(123)),
            ..Default::default()
        }),
        ..Default::default()
    });
    config.profiles.insert("test".to_string(), anthropic_bridge::config::Profile {
        rules: vec![anthropic_bridge::config::Rule {
            pattern: "test".to_string(),
            target: "override_auto".to_string(),
            match_features: vec![],
            reasoning_target: None,
        }],
    });
    config.current_profile = "test".to_string();
    config.upstream.base_url = Some(mock_server.uri());

    let state = Arc::new(AppState {
        config,
        client: reqwest::Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("k".to_string()),
        verbose: true,
        tool_verbose: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    app.oneshot(Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1000
        }).to_string()))
        .unwrap())
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    
    assert_eq!(r.max_tokens, Some(123));
}

#[tokio::test]
async fn test_override_max_tokens_human_readable() {
    let mock_server = MockServer::start().await;
    
    let mut config = Config::default();
    config.models.insert("human_readable".to_string(), ModelConfig {
        api_model_id: Some("test".to_string()),
        override_max_tokens: Some(json!("64k")),
        ..Default::default()
    });
    config.profiles.insert("test".to_string(), anthropic_bridge::config::Profile {
        rules: vec![anthropic_bridge::config::Rule {
            pattern: "test".to_string(),
            target: "human_readable".to_string(),
            match_features: vec![],
            reasoning_target: None,
        }],
    });
    config.current_profile = "test".to_string();
    config.upstream.base_url = Some(mock_server.uri());

    let state = Arc::new(AppState {
        config,
        client: reqwest::Client::new(),
        base_url: mock_server.uri(),
        api_key: Some("k".to_string()),
        verbose: true,
        tool_verbose: false,
        tools_reported: AtomicBool::new(false),
    });
    let app = create_router(state);

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    app.oneshot(Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1000
        }).to_string()))
        .unwrap())
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    let r: OpenAIChatCompletionRequest = serde_json::from_slice(&reqs[0].body).unwrap();
    
    assert_eq!(r.max_tokens, Some(64000));
}
