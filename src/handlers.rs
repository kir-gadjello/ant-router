use crate::config::Config;
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
use tracing::{debug, error, info};

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

    // 1. Resolve Model
    let thinking_enabled = payload.thinking.is_some();
    let upstream_model = state.config.resolve_model(&payload.model, thinking_enabled);

    info!(
        "Request for model '{}' (thinking={}) resolved to '{}'",
        payload.model, thinking_enabled, upstream_model
    );

    // 2. Transform Request
    let openai_req = match convert_request(payload.clone(), upstream_model.clone()) {
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

    // 3. Prepare Upstream Request
    let url = format!("{}/v1/chat/completions", state.base_url);
    let mut req_builder = state.client.post(&url).json(&openai_req);

    if let Some(key) = &state.api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
    }

    // Pass through some headers if needed, or just standard ones?
    // Spec doesn't specify passing headers, but usually good practice to pass Trace IDs.
    // For now, minimal headers.

    // Debug logging
    if env::var("DEBUG").is_ok() {
        debug!("Upstream Request: {:?}", openai_req);
    }

    // 4. Send Request
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
        // Try to parse error as JSON to return cleanly, else text
        let error_body =
            serde_json::from_str::<Value>(&error_text).unwrap_or(json!({"error": error_text}));
        return (status, Json(error_body)).into_response();
    }

    // 5. Handle Response
    if payload.stream == Some(true) {
        let stream = response.bytes_stream();

        let openai_stream = parse_sse_stream(stream);
        let anthropic_stream = convert_stream(openai_stream);

        // Box::pin to ensure Unpin
        let sse_stream = Box::pin(anthropic_stream.map(|res| {
            match res {
                Ok(event) => {
                    // Extract event type for the `event: ` line
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
                    // Safe to convert to string here as we expect valid text between boundaries
                    // But use lossy to be safe against upstream malformed bytes
                    let s = String::from_utf8_lossy(&packet);

                    for line in s.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            let trimmed = data.trim();
                            if trimmed == "[DONE]" {
                                break;
                            }
                            // Skip empty or keepalive
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
