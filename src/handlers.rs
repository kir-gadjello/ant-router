use crate::config::Config;
use crate::logging::log_request;
use crate::protocol::*;
use crate::transformer::{convert_request, convert_response, convert_stream};
use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use bytes::BytesMut;
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::env;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

pub struct AppState {
    pub config: Config,
    pub client: Client,
    pub base_url: String,
    pub api_key: Option<String>,
}

pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "healthy"})))
}

pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(payload): Json<AnthropicMessageRequest>,
) -> impl IntoResponse {
    let _start = std::time::Instant::now();

    // 0. Logging
    if state.config.log_enabled {
        let log_path = state.config.get_log_path();
        // Log asynchronously/non-blocking ideally, but for now blocking IO is simple and acceptable for this task
        // as per "I trust your engineering intuition".
        // To be strictly async, we'd spawn_blocking.
        let payload_clone = payload.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = log_request(&payload_clone, &log_path) {
                warn!("Failed to log request: {}", e);
            }
        });
    }

    // 1. Detect Vision Requirements
    let has_images = payload.messages.iter().any(|m| match &m.content {
        AnthropicMessageContent::Blocks(blocks) => blocks
            .iter()
            .any(|b| matches!(b, AnthropicContentBlock::Image { .. })),
        _ => false,
    });

    let thinking_enabled = payload.thinking.is_some();

    // 2. Resolve Model ID (The logical ID from config, not yet the wire ID)
    // We first check if we need to route based on vision capabilities
    let target_model_id = if has_images {
        if thinking_enabled {
            state
                .config
                .ant_vision_reasoning_model
                .as_deref()
                .or(state.config.ant_vision_model.as_deref())
                .unwrap_or(&payload.model) // Fallback to requested model
        } else {
            state
                .config
                .ant_vision_model
                .as_deref()
                .unwrap_or(&payload.model)
        }
    } else {
        // Standard resolution logic
        if let Some(_model_conf) = state.config.resolve_model_alias(&payload.model) {
            // It's a UMCP model
            &payload.model
        } else {
            // It might be a pattern match from legacy profiles, or just pass-through
            &payload.model
        }
    }
    .to_string();

    // 3. Resolve Final Configuration (Wire ID + Params)
    // Try to find the UMCP configuration for the target ID
    let (wire_model, api_params) =
        if let Some(model_conf) = state.config.resolve_model_alias(&target_model_id) {
            // Found a UMCP model config
            let wire_id = state.config.get_wire_model_id(model_conf);
            (wire_id, model_conf.api_params.clone())
        } else {
            // Fallback to legacy resolution
            let wire_id = state
                .config
                .resolve_model(&target_model_id, thinking_enabled);
            (wire_id, None)
        };

    info!(
        "Request for '{}' (thinking={}, vision={}) -> UMCP ID '{}' -> Wire '{}'",
        payload.model, thinking_enabled, has_images, target_model_id, wire_model
    );

    // 4. Transform Request
    let openai_req = match convert_request(payload.clone(), wire_model.clone()) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to convert request: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Invalid request: {}", e)})),
            )
                .into_response();
        }
    };

    // 5. Prepare Upstream Request
    let url = format!("{}/v1/chat/completions", state.base_url);

    let mut req_builder = state.client.post(&url);

    if let Some(key) = &state.api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }

    // Apply headers from UMCP params
    if let Some(params) = &api_params {
        for (k, v) in &params.headers {
            req_builder = req_builder.header(k, v);
        }
    }

    // Merge Body
    let mut final_body = serde_json::to_value(&openai_req).unwrap();
    if let Some(params) = &api_params {
        if !params.extra_body.is_empty() {
            if let Value::Object(ref mut map) = final_body {
                for (k, v) in &params.extra_body {
                    map.insert(k.clone(), v.clone());
                }
            }
        }
    }

    req_builder = req_builder.json(&final_body);

    // Debug logging
    if env::var("DEBUG").is_ok() {
        debug!("Upstream Request Body: {:?}", final_body);
    }

    // 6. Send Request
    let response = match req_builder.send().await {
        Ok(res) => res,
        Err(e) => {
            error!("Upstream request failed: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        error!("Upstream error {}: {}", status, error_text);
        let error_body =
            serde_json::from_str::<Value>(&error_text).unwrap_or(json!({"error": error_text}));
        return (status, Json(error_body)).into_response();
    }

    // 7. Handle Response
    if payload.stream == Some(true) {
        let stream = response.bytes_stream();

        let openai_stream = parse_sse_stream(stream);
        let anthropic_stream = convert_stream(openai_stream);

        // Box::pin to ensure Unpin
        let sse_stream = Box::pin(anthropic_stream.map(|res| {
            match res {
                Ok(event) => {
                    let event_type = get_event_type(&event);
                    Event::default().event(event_type).json_data(event)
                }
                Err(e) => Event::default()
                    .event("error")
                    .json_data(json!({"error": e.to_string()})),
            }
        }));

        let sse = Sse::new(sse_stream).keep_alive(KeepAlive::default());

        sse.into_response()
    } else {
        // Non-streaming
        let openai_resp: OpenAIChatCompletionResponse = match response.json().await {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to parse upstream response: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Failed to parse upstream response"})),
                )
                    .into_response();
            }
        };

        if env::var("DEBUG").is_ok() {
            debug!("Upstream Response: {:?}", openai_resp);
        }

        match convert_response(openai_resp) {
            Ok(anthropic_resp) => (StatusCode::OK, Json(anthropic_resp)).into_response(),
            Err(e) => {
                error!("Failed to convert response: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    }
}

fn get_event_type(event: &AnthropicStreamEvent) -> &'static str {
    match event {
        AnthropicStreamEvent::MessageStart { .. } => "message_start",
        AnthropicStreamEvent::ContentBlockStart { .. } => "content_block_start",
        AnthropicStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        AnthropicStreamEvent::ContentBlockStop { .. } => "content_block_stop",
        AnthropicStreamEvent::MessageDelta { .. } => "message_delta",
        AnthropicStreamEvent::MessageStop => "message_stop",
        AnthropicStreamEvent::Ping => "ping",
        AnthropicStreamEvent::Error { .. } => "error",
    }
}

// Minimal SSE Parser helper
fn parse_sse_stream<S>(
    byte_stream: S,
) -> impl futures::Stream<Item = Result<OpenAIChatCompletionChunk, anyhow::Error>>
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    async_stream::try_stream! {
        let mut buffer = BytesMut::new();
        let mut stream = byte_stream;

        while let Some(chunk_res) = stream.next().await {
            let chunk = chunk_res?;
            buffer.extend_from_slice(&chunk);

            loop {
                // Find double newline to separate events
                let mut found_idx = None;
                // Simple search
                for i in 0..buffer.len().saturating_sub(1) {
                    if buffer[i] == b'\n' && buffer[i+1] == b'\n' {
                        found_idx = Some(i);
                        break;
                    }
                }

                if let Some(idx) = found_idx {
                    let packet = buffer.split_to(idx + 2);
                    let s = String::from_utf8_lossy(&packet);

                    for line in s.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            let trimmed = data.trim();
                            if trimmed == "[DONE]" {
                                break;
                            }
                            if trimmed.is_empty() {
                                continue;
                            }

                            if let Ok(chunk) = serde_json::from_str::<OpenAIChatCompletionChunk>(trimmed) {
                                yield chunk;
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;

    #[tokio::test]
    async fn test_parse_sse_stream_split_utf8() {
        // "Hello 🌍" -> 🌍 is 4 bytes: F0 9F 8C 8D
        // We will split the bytes of 🌍 across chunks

        let chunk1 = Bytes::from("data: {\"id\":\"1\",\"object\":\"chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"}}]}\n\n");
        // Part of next message: "data: ... content: ... " then start of 🌍
        let part1 = "data: {\"id\":\"1\",\"object\":\"chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";

        // 🌍 bytes
        let earth = "🌍";
        let earth_bytes = earth.as_bytes(); // [240, 159, 140, 141]

        let mut chunk2 = BytesMut::new();
        chunk2.extend_from_slice(part1.as_bytes());
        chunk2.extend_from_slice(&earth_bytes[0..2]); // F0 9F

        let mut chunk3 = BytesMut::new();
        chunk3.extend_from_slice(&earth_bytes[2..4]); // 8C 8D
        chunk3.extend_from_slice(b"\"}}]}\n\n");

        let chunk4 = Bytes::from("data: [DONE]\n\n");

        let stream = stream::iter(vec![
            Ok(chunk1),
            Ok(chunk2.freeze()),
            Ok(chunk3.freeze()),
            Ok(chunk4),
        ]);

        let mut parsed_stream = Box::pin(parse_sse_stream(stream));

        let c1 = parsed_stream.next().await.unwrap().unwrap();
        assert_eq!(c1.choices[0].delta.content.as_deref(), Some("Hello "));

        // The second message should be fully assembled before parsing
        let c2 = parsed_stream.next().await.unwrap().unwrap();
        assert_eq!(c2.choices[0].delta.content.as_deref(), Some("🌍"));

        assert!(parsed_stream.next().await.is_none());
    }
}
