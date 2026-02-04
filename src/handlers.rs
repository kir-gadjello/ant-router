use crate::config::{Config, glob_to_regex};
use crate::logging::{log_request, record_interaction};
use crate::middleware::Middleware;
use crate::protocol::*;
use crate::transformer::{convert_request, convert_response, convert_stream, record_stream};
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
use regex::Regex;
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

pub struct AppState {
    pub config: Config,
    pub client: Client,
    pub base_url: String,
    pub api_key: Option<String>,
    pub verbose: bool,
    pub tool_verbose: bool,
    pub record: bool,
    pub tools_reported: AtomicBool,
    pub middlewares: Vec<Box<dyn Middleware>>,
}

pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "healthy"})))
}

pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(mut payload): Json<AnthropicMessageRequest>,
) -> impl IntoResponse {
    let _start = std::time::Instant::now();

    // Middleware: on_request
    for m in &state.middlewares {
        if let Err(e) = m.on_request(&mut payload) {
            error!("Middleware request error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    }

    // 0. Logging
    if state.config.log_enabled {
        let log_path = state.config.get_log_path();
        let payload_clone = payload.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = log_request(&payload_clone, &log_path) {
                warn!("Failed to log request: {}", e);
            }
        });
    }

    // 1. Detect Features
    let has_images = payload.messages.iter().any(|m| match &m.content {
        AnthropicMessageContent::Blocks(blocks) => blocks
            .iter()
            .any(|b| matches!(b, AnthropicContentBlock::Image { .. })),
        _ => false,
    });

    let thinking_enabled = payload.thinking.is_some();

    let mut request_features = Vec::new();
    if has_images {
        request_features.push("vision");
    }
    if thinking_enabled {
        request_features.push("reasoning");
    }

    // 2. Resolve Model Configuration
    let profile_name = &state.config.current_profile;
    let profile = match state.config.profiles.get(profile_name) {
        Some(p) => p,
        None => {
            error!("Current profile '{}' not found in config", profile_name);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Profile '{}' not found", profile_name)})),
            ).into_response();
        }
    };

    let target_logical_id = resolve_via_rules(profile, &payload.model, &request_features);

    // Lookup ModelConfig
    let model_conf = state.config.models.get(&target_logical_id);
    let (wire_model, api_params) = if let Some(conf) = model_conf {
        (state.config.get_wire_model_id(conf), conf.api_params.clone())
    } else {
        warn!("Logical model ID '{}' resolved but not found in models definition", target_logical_id);
        (target_logical_id.clone(), None)
    };

    // 3. Check `no_ant` flag
    if state.config.no_ant && wire_model.to_lowercase().contains("anthropic") {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "Requests to Anthropic models are forbidden by configuration"})),
        ).into_response();
    }

    info!(
        "Request for '{}' (features={:?}) -> Logical '{}' -> Wire '{}'",
        payload.model, request_features, target_logical_id, wire_model
    );

    // 4. Transform Request
    let (openai_req, report) = match convert_request(payload.clone(), wire_model.clone(), model_conf) {
        Ok(res) => res,
        Err(e) => {
            error!("Failed to convert request: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Invalid request: {}", e)})),
            ).into_response();
        }
    };

    // Tool Verbose Reporting (Once per session)
    if state.tool_verbose {
        report_tools_once(&state, &report);
    }

    // 5. Prepare Upstream Request
    let url = format!("{}/v1/chat/completions", state.base_url);
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

    // --- VERBOSE LOGGING START ---
    if state.verbose {
        let mut log_body = final_body.clone();
        truncate_long_strings(&mut log_body);
        info!("Upstream Request Body: {}", serde_json::to_string_pretty(&log_body).unwrap_or_default());
    }
    // --- VERBOSE LOGGING END ---

    // 6. Send Request (with Retry)
    let max_retries = api_params.as_ref().and_then(|p| p.retry.as_ref()).map(|r| r.max_retries).unwrap_or(0);
    let backoff = api_params.as_ref().and_then(|p| p.retry.as_ref()).map(|r| r.backoff_ms).unwrap_or(500);

    let mut attempts = 0;
    let mut context_length_retries = 0;
    let max_context_retries = 1;

    let response = loop {
        attempts += 1;
        
        let mut attempt_builder = state.client.post(&url);
        if let Some(key) = &state.api_key {
            attempt_builder = attempt_builder.header("Authorization", format!("Bearer {}", key));
        }
        if let Some(params) = &api_params {
            for (k, v) in &params.headers {
                attempt_builder = attempt_builder.header(k, v);
            }
        }
        attempt_builder = attempt_builder.json(&final_body);

        match attempt_builder.send().await {
            Ok(res) => {
                let status = res.status();
                if status.is_server_error() && attempts <= max_retries {
                    warn!("Upstream server error {}, retrying ({}/{})", status, attempts, max_retries);
                    tokio::time::sleep(Duration::from_millis(backoff * (2_u64.pow(attempts - 1)))).await;
                    continue;
                }
                
                if status == StatusCode::BAD_REQUEST && context_length_retries < max_context_retries {
                    let error_bytes = res.bytes().await.unwrap_or_default();
                    let error_text = String::from_utf8_lossy(&error_bytes);
                    
                    // Robust regex to parse OpenRouter context length error across newlines and multiple input types
                    let re = Regex::new(r"(?s)maximum context length is (\d+).*?requested about (\d+).*?(\d+) in the output").unwrap();
                    if let Some(caps) = re.captures(&error_text) {
                        if let (Some(max_ctx), Some(total_req), Some(output_req)) = (caps.get(1), caps.get(2), caps.get(3)) {
                            if let (Ok(max_ctx_val), Ok(total_val), Ok(output_val)) = (
                                max_ctx.as_str().parse::<u32>(), 
                                total_req.as_str().parse::<u32>(),
                                output_req.as_str().parse::<u32>()
                            ) {
                                let input_val = total_val.saturating_sub(output_val);
                                info!("Caught context length error. Max: {}, Total Requested: {}, Output Requested: {} (Input: {}). Adjusting max_tokens and retrying.", 
                                    max_ctx_val, total_val, output_val, input_val);
                                
                                let safety_margin = 100;
                                let available = max_ctx_val.saturating_sub(input_val).saturating_sub(safety_margin);
                                if available > 0 {
                                    if let Value::Object(ref mut map) = final_body {
                                        map.insert("max_tokens".to_string(), json!(available));
                                    }
                                    context_length_retries += 1;
                                    continue;
                                }
                            }
                        }
                    }
                    error!("Upstream 400 Bad Request: {}", error_text);
                    let error_body: Value = serde_json::from_slice(&error_bytes).unwrap_or(json!({"error": error_text}));
                    return (StatusCode::BAD_REQUEST, Json(error_body)).into_response();
                }
                
                break res;
            }
            Err(e) => {
                if attempts <= max_retries {
                    warn!("Upstream connection failed: {}, retrying ({}/{})", e, attempts, max_retries);
                    tokio::time::sleep(Duration::from_millis(backoff * (2_u64.pow(attempts - 1)))).await;
                    continue;
                }
                return (StatusCode::BAD_GATEWAY, Json(json!({"error": e.to_string()}))).into_response();
            }
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        error!("Upstream error {}: {}", status, error_text);
        let error_body = serde_json::from_str::<Value>(&error_text).unwrap_or(json!({"error": error_text}));
        return (status, Json(error_body)).into_response();
    }

    // 7. Handle Response
    if payload.stream == Some(true) {
        let stream = response.bytes_stream();
        let openai_stream = parse_sse_stream(stream);
        let anthropic_stream = convert_stream(openai_stream);

        let mut final_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<AnthropicStreamEvent, anyhow::Error>> + Send>,
        > = Box::pin(anthropic_stream);

        // Middleware: transform_stream
        for m in &state.middlewares {
            final_stream = m.transform_stream(final_stream);
        }

        let recorded_stream = if state.record {
            let req_clone = payload.clone();
            Box::pin(record_stream(final_stream, move |full_resp| {
                let resp_value = serde_json::to_value(&full_resp)
                    .unwrap_or(json!({"error": "Failed to serialize response"}));
                if let Err(e) = record_interaction(&req_clone, &resp_value) {
                    warn!("Failed to record streaming interaction: {}", e);
                }
            }))
                as std::pin::Pin<
                    Box<dyn futures::Stream<Item = Result<AnthropicStreamEvent, anyhow::Error>> + Send>,
                >
        } else {
            final_stream
        };

        let sse_stream = Box::pin(recorded_stream.map(|res: Result<AnthropicStreamEvent, anyhow::Error>| {
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

        Sse::new(sse_stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let openai_resp: OpenAIChatCompletionResponse = match response.json().await {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to parse upstream response: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "Failed to parse upstream response"}))).into_response();
            }
        };

        if state.verbose {
            let mut log_resp = serde_json::to_value(&openai_resp).unwrap_or_default();
            truncate_long_strings(&mut log_resp);
            info!("Upstream Response Body: {}", serde_json::to_string_pretty(&log_resp).unwrap_or_default());
        }

        match convert_response(openai_resp) {
            Ok(mut anthropic_resp) => {
                // Middleware: on_response
                for m in &state.middlewares {
                    if let Err(e) = m.on_response(&mut anthropic_resp) {
                        error!("Middleware response error: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({"error": e.to_string()})),
                        )
                            .into_response();
                    }
                }

                if state.record {
                    let req_clone = payload.clone();
                    let resp_value = serde_json::to_value(&anthropic_resp)
                        .unwrap_or(json!({"error": "Failed to serialize response"}));
                    tokio::task::spawn_blocking(move || {
                        if let Err(e) = record_interaction(&req_clone, &resp_value) {
                            warn!("Failed to record interaction: {}", e);
                        }
                    });
                }
                (StatusCode::OK, Json(anthropic_resp)).into_response()
            }
            Err(e) => {
                error!("Failed to convert response: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response()
            }
        }
    }
}

fn report_tools_once(state: &AppState, report: &crate::transformer::request::PreprocessReport) {
    if !state.tools_reported.swap(true, Ordering::Relaxed) {
        info!("--- Tool Sanitization Report (First Request) ---");
        if report.sanitized_tool_ids.is_empty() {
            info!("No tools were removed during sanitization.");
        } else {
            info!("The following tool IDs were removed due to invalid definitions (empty names):");
            for id in &report.sanitized_tool_ids {
                info!(" - {}", id);
            }
        }
        
        if report.passed_tool_ids.is_empty() {
            info!("No tools passed sanitization (none provided or all removed).");
        } else {
            info!("The following tool IDs passed sanitization:");
            for id in &report.passed_tool_ids {
                info!(" - {}", id);
            }
        }
        info!("----------------------------------------------");
    }
}

// Updated resolver using match_features
fn resolve_via_rules(
    profile: &crate::config::Profile,
    input_model: &str,
    request_features: &[&str],
) -> String {
    for rule in &profile.rules {
        if let Ok(regex) = glob_to_regex(&rule.pattern) {
            if regex.is_match(input_model) {
                // Check if rule requires features
                // Logic: A rule matches if ALL its required features are present in the request.
                // Or: A rule matches if the request has features AND the rule matches them?
                // The prompt says: "match_features: [vision, vision_reasoning]".
                // This implies "Match this rule IF the request has these features".

                let rule_requires_features = !rule.match_features.is_empty();

                if rule_requires_features {
                    // Check if request has AT LEAST ONE of the matching features?
                    // Or ALL? Usually match lists are "if request has any of these".
                    // E.g. match_features: [vision] -> matches if request has vision.
                    // But if match_features: [vision, reasoning] -> matches if request has vision OR reasoning?
                    // Or implies a combined capability?
                    // Let's assume "Match if request has ANY of the listed features".
                    // So if request has "vision", and rule has "vision", it matches.
                    // If rule has "vision", and request has "reasoning", it does NOT match.

                    let mut feature_match = false;
                    for rf in request_features {
                        if rule.match_features.iter().any(|f| f == *rf) {
                            feature_match = true;
                            break;
                        }
                    }

                    // Special case: "vision_reasoning" feature in rule might match if request has BOTH vision and reasoning?
                    // Simplified: just match simple strings.

                    if feature_match {
                        return rule.target.clone();
                    }
                } else {
                    // Rule has no feature constraints. It matches general requests.
                    // BUT: we should prefer feature-specific rules.
                    // Since we iterate in order, we assume feature-specific rules come FIRST.
                    // If we are here, it means we matched pattern but didn't check features.
                    // If this rule is a catch-all (no features), it matches.
                    return rule.target.clone();
                }
            }
        }
    }
    // Fallback
    input_model.to_string()
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
                let mut found_idx = None;
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

fn truncate_long_strings(v: &mut Value) {
    match v {
        Value::String(s) => {
            if s.len() > 100 {
                let mut truncated = s.chars().take(100).collect::<String>();
                truncated.push_str("... [truncated]");
                *s = truncated;
            }
        }
        Value::Array(arr) => {
            for item in arr {
                truncate_long_strings(item);
            }
        }
        Value::Object(map) => {
            for (_, val) in map {
                truncate_long_strings(val);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_parse_sse_stream_split_utf8() {
        let chunk1 = Bytes::from("data: {\"id\":\"1\",\"object\":\"chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"}}]}\n\n");
        let part1 = "data: {\"id\":\"1\",\"object\":\"chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";
        let earth = "🌍";
        let earth_bytes = earth.as_bytes();

        let mut chunk2 = BytesMut::new();
        chunk2.extend_from_slice(part1.as_bytes());
        chunk2.extend_from_slice(&earth_bytes[0..2]);

        let mut chunk3 = BytesMut::new();
        chunk3.extend_from_slice(&earth_bytes[2..4]);
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

        let c2 = parsed_stream.next().await.unwrap().unwrap();
        assert_eq!(c2.choices[0].delta.content.as_deref(), Some("🌍"));

        assert!(parsed_stream.next().await.is_none());
    }
}
