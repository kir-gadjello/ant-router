use crate::config::{
    glob_to_regex, merge_preprocess, ApiParams, Config, SystemPromptOp, SystemPromptRule,
    ToolFilterConfig,
};
use crate::logging::{log_request, record_interaction};
use crate::middleware::system_prompt::SystemPromptPatcherMiddleware;
use crate::middleware::tool_enforcer::ToolEnforcerMiddleware;
use crate::middleware::tool_filter::ToolFilterMiddleware;
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
    pub debug_tools: bool,
    pub record: bool,
    pub tools_reported: AtomicBool,
}

pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "healthy"})))
}

pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(mut payload): Json<AnthropicMessageRequest>,
) -> impl IntoResponse {
    // Check for override
    let (target_profile_name, target_logical_id_override) = check_overrides(&payload.model);

    let profile_name = target_profile_name
        .as_deref()
        .unwrap_or(&state.config.current_profile);

    let profile = match state.config.profiles.get(profile_name) {
        Some(p) => p,
        None => {
            error!("Current profile '{}' not found in config", profile_name);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Profile '{}' not found", profile_name)})),
            )
                .into_response();
        }
    };

    // Construct Middlewares
    let mut middlewares: Vec<Box<dyn Middleware>> = Vec::new();

    let tool_filter_config = profile
        .tool_filters
        .clone()
        .or_else(|| state.config.tool_filters.clone());
    middlewares.push(Box::new(ToolFilterMiddleware::new(tool_filter_config)));

    // Task T3: Profile specific system prompts
    let system_prompts = profile
        .system_prompts
        .clone()
        .unwrap_or_else(|| state.config.system_prompts.clone());
    middlewares.push(Box::new(SystemPromptPatcherMiddleware::new(system_prompts)));

    let enable_exit_tool = profile
        .enable_exit_tool
        .unwrap_or(state.config.enable_exit_tool);

    if enable_exit_tool {
        middlewares.push(Box::new(ToolEnforcerMiddleware::new()));
    }

    // Middleware: on_request
    for m in &middlewares {
        if let Err(e) = m.on_request(&mut payload) {
            error!("Middleware request error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    }

    // Logging
    if state.config.log_enabled {
        let log_path = state.config.get_log_path();
        let payload_clone = payload.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = log_request(&payload_clone, &log_path) {
                warn!("Failed to log request: {}", e);
            }
        });
    }

    // Feature Detection
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

    // Model Resolution
    let target_logical_id = if let Some(id) = target_logical_id_override {
        id
    } else {
        resolve_via_rules(profile, &payload.model, &request_features)
    };

    // Lookup ModelConfig
    let base_model_conf = state.config.models.get(&target_logical_id);

    // Merge Profile Preprocess with Model Preprocess (Profile overrides Model)
    let mut model_conf = base_model_conf.cloned().unwrap_or_default();

    if let Some(profile_preprocess) = &profile.preprocess {
        model_conf.preprocess = merge_preprocess(model_conf.preprocess, Some(profile_preprocess.clone()));
    }

    let (wire_model, api_params) = if base_model_conf.is_some() {
        (
            state.config.get_wire_model_id(&model_conf),
            model_conf.api_params.clone(),
        )
    } else {
        warn!(
            "Logical model ID '{}' resolved but not found in models definition",
            target_logical_id
        );
        (target_logical_id.clone(), None)
    };

    if state.config.no_ant && wire_model.to_lowercase().contains("anthropic") {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "Requests to Anthropic models are forbidden by configuration"})),
        )
            .into_response();
    }

    info!(
        "Request for '{}' (features={:?}) -> Logical '{}' -> Wire '{}'",
        payload.model, request_features, target_logical_id, wire_model
    );

    // Convert Request
    let (openai_req, report) =
        match convert_request(payload.clone(), wire_model.clone(), Some(&model_conf), state.debug_tools) {
            Ok(res) => res,
            Err(e) => {
                error!("Failed to convert request: {}", e);
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": format!("Invalid request: {}", e)})),
                )
                    .into_response();
            }
        };

    if state.tool_verbose {
        report_tools_once(&state, &report);
    }

    // Prepare Body
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

    // Execute
    let url = format!("{}/v1/chat/completions", state.base_url);
    let response = match execute_upstream_request(&state, &url, final_body, &api_params).await {
        Ok(res) => res,
        Err(e) => return e.into_response(),
    };

    // Handle Response
    if payload.stream == Some(true) {
        let stream = response.bytes_stream();
        let openai_stream = parse_sse_stream(stream);
        let anthropic_stream = convert_stream(openai_stream, state.debug_tools);

        let mut final_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<AnthropicStreamEvent, anyhow::Error>> + Send>,
        > = Box::pin(anthropic_stream);

        // Record interaction if enabled
        if state.record {
             let req_clone = payload.clone();
             final_stream = Box::pin(record_stream(final_stream, move |accumulated_resp| {
                 let resp_value = serde_json::to_value(&accumulated_resp).unwrap_or(json!({"error": "Failed to serialize response"}));
                 tokio::task::spawn_blocking(move || {
                     if let Err(e) = record_interaction(&req_clone, &resp_value) {
                         warn!("Failed to record interaction: {}", e);
                     }
                 });
             }));
        }

        let sse_stream = Box::pin(final_stream.map(|res| {
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

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
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

        if state.verbose {
            let mut log_resp = serde_json::to_value(&openai_resp).unwrap_or_default();
            truncate_long_strings(&mut log_resp);
            info!(
                "Upstream Response Body: {}",
                serde_json::to_string_pretty(&log_resp).unwrap_or_default()
            );
        }

        match convert_response(openai_resp, Some(&model_conf), state.debug_tools) {
            Ok(mut anthropic_resp) => {
                for m in &middlewares {
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
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    }
}

pub async fn handle_openai_chat(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(mut payload): Json<OpenAIChatCompletionRequest>,
) -> impl IntoResponse {
    // 1. Resolve Profile & Model (with Overrides)
    let (target_profile_name, target_logical_id_override) = check_overrides(&payload.model);

    let profile_name = target_profile_name
        .as_deref()
        .unwrap_or(&state.config.current_profile);

    let profile = match state.config.profiles.get(profile_name) {
        Some(p) => p,
        None => {
            error!("Current profile '{}' not found in config", profile_name);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Profile '{}' not found", profile_name)})),
            )
                .into_response();
        }
    };

    // 2. Middleware Helpers
    // System Prompts
    let system_prompts = profile
        .system_prompts
        .clone()
        .unwrap_or_else(|| state.config.system_prompts.clone());
    apply_openai_system_prompt_patch(&mut payload, &system_prompts);

    // Tool Filtering
    let tool_filter_config = profile
        .tool_filters
        .clone()
        .or_else(|| state.config.tool_filters.clone());
    apply_openai_tool_filter(&mut payload, &tool_filter_config);

    // Feature Detection (OpenAI specific)
    let mut request_features = Vec::new();
    // Check for images
    for msg in &payload.messages {
        if let Some(OpenAIContent::Array(parts)) = &msg.content {
            if parts
                .iter()
                .any(|p| matches!(p, OpenAIContentPart::ImageUrl { .. }))
            {
                request_features.push("vision");
                break;
            }
        }
    }


    // Model Resolution
    let target_logical_id = if let Some(id) = target_logical_id_override {
        id
    } else {
        resolve_via_rules(profile, &payload.model, &request_features)
    };

    // Lookup ModelConfig
    let base_model_conf = state.config.models.get(&target_logical_id);

    // Merge Profile Preprocess with Model Preprocess (Profile overrides Model)
    let mut model_conf = base_model_conf.cloned().unwrap_or_default();

    if let Some(profile_preprocess) = &profile.preprocess {
        // Merge strategy: Profile overrides Model
        model_conf.preprocess = merge_preprocess(model_conf.preprocess, Some(profile_preprocess.clone()));
    }

    let (wire_model, api_params) = if base_model_conf.is_some() {
        (
            state.config.get_wire_model_id(&model_conf),
            model_conf.api_params.clone(),
        )
    } else {
        warn!(
            "Logical model ID '{}' resolved but not found in models definition",
            target_logical_id
        );
        (target_logical_id.clone(), None)
    };

    if state.config.no_ant && wire_model.to_lowercase().contains("anthropic") {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "Requests to Anthropic models are forbidden by configuration"})),
        )
            .into_response();
    }

    info!(
        "OpenAI Request for '{}' (features={:?}) -> Logical '{}' -> Wire '{}'",
        payload.model, request_features, target_logical_id, wire_model
    );

    // Update Model in Payload
    payload.model = wire_model;

    // Prepare Upstream
    let mut final_body = serde_json::to_value(&payload).unwrap();
    if let Some(params) = &api_params {
        if !params.extra_body.is_empty() {
            if let Value::Object(ref mut map) = final_body {
                for (k, v) in &params.extra_body {
                    map.insert(k.clone(), v.clone());
                }
            }
        }
    }

    let url = format!("{}/v1/chat/completions", state.base_url);
    let response = match execute_upstream_request(&state, &url, final_body, &api_params).await {
        Ok(res) => res,
        Err(e) => return e.into_response(),
    };

    if payload.stream == Some(true) {
        let stream = response.bytes_stream();
        let openai_stream = parse_sse_stream(stream);

        // Pass-through stream
        // This endpoint acts as a proxy for OpenAI clients, so we return OpenAI chunks directly.
        let sse_stream = Box::pin(openai_stream.map(|res| {
            match res {
                Ok(chunk) => Event::default().json_data(chunk),
                Err(e) => Event::default()
                    .event("error")
                    .json_data(json!({"error": e.to_string()})),
            }
        }));

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let resp_body_bytes = match response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        };

        // We can parse it to verify or log
        let resp_json: Value = match serde_json::from_slice(&resp_body_bytes) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        };

        if state.verbose {
            let mut log_resp = resp_json.clone();
            truncate_long_strings(&mut log_resp);
            info!(
                "Upstream Response Body: {}",
                serde_json::to_string_pretty(&log_resp).unwrap_or_default()
            );
        }

        (StatusCode::OK, Json(resp_json)).into_response()
    }
}

async fn execute_upstream_request(
    state: &Arc<AppState>,
    url: &str,
    mut final_body: Value,
    api_params: &Option<ApiParams>,
) -> Result<reqwest::Response, (StatusCode, Json<Value>)> {
    if state.verbose {
        let mut log_body = final_body.clone();
        truncate_long_strings(&mut log_body);
        info!(
            "Upstream Request Body: {}",
            serde_json::to_string_pretty(&log_body).unwrap_or_default()
        );
    }

    let max_retries = api_params
        .as_ref()
        .and_then(|p| p.retry.as_ref())
        .map(|r| r.max_retries)
        .unwrap_or(0);
    let backoff = api_params
        .as_ref()
        .and_then(|p| p.retry.as_ref())
        .map(|r| r.backoff_ms)
        .unwrap_or(500);

    let mut attempts = 0;
    let mut context_length_retries = 0;
    let max_context_retries = 1;
    let re = Regex::new(
        r"(?s)maximum context length is (\d+).*?requested about (\d+).*?(\d+) in the output",
    )
    .unwrap();

    loop {
        attempts += 1;

        let mut attempt_builder = state.client.post(url);
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
                    warn!(
                        "Upstream server error {}, retrying ({}/{})",
                        status, attempts, max_retries
                    );
                    tokio::time::sleep(Duration::from_millis(backoff * (2_u64.pow(attempts - 1))))
                        .await;
                    continue;
                }

                if status == StatusCode::BAD_REQUEST && context_length_retries < max_context_retries
                {
                    let error_bytes = res.bytes().await.unwrap_or_default();
                    let error_text = String::from_utf8_lossy(&error_bytes);

                    if let Some(caps) = re.captures(&error_text) {
                        if let (Some(max_ctx), Some(total_req), Some(output_req)) =
                            (caps.get(1), caps.get(2), caps.get(3))
                        {
                            if let (Ok(max_ctx_val), Ok(total_val), Ok(output_val)) = (
                                max_ctx.as_str().parse::<u32>(),
                                total_req.as_str().parse::<u32>(),
                                output_req.as_str().parse::<u32>(),
                            ) {
                                let input_val = total_val.saturating_sub(output_val);
                                info!("Caught context length error. Max: {}, Total Requested: {}, Output Requested: {} (Input: {}). Adjusting max_tokens and retrying.", 
                                    max_ctx_val, total_val, output_val, input_val);

                                let safety_margin = 100;
                                let available = max_ctx_val
                                    .saturating_sub(input_val)
                                    .saturating_sub(safety_margin);
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
                    let error_body: Value = serde_json::from_slice(&error_bytes)
                        .unwrap_or(json!({"error": error_text}));
                    return Err((StatusCode::BAD_REQUEST, Json(error_body)));
                }

                if !status.is_success() {
                    let error_text = res.text().await.unwrap_or_default();
                    error!("Upstream error {}: {}", status, error_text);
                    let error_body = serde_json::from_str::<Value>(&error_text)
                        .unwrap_or(json!({"error": error_text}));
                    return Err((status, Json(error_body)));
                }

                return Ok(res);
            }
            Err(e) => {
                if attempts <= max_retries {
                    warn!(
                        "Upstream connection failed: {}, retrying ({}/{})",
                        e, attempts, max_retries
                    );
                    tokio::time::sleep(Duration::from_millis(backoff * (2_u64.pow(attempts - 1))))
                        .await;
                    continue;
                }
                return Err((
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": e.to_string()})),
                ));
            }
        }
    }
}

// Override Logic
fn check_overrides(model: &str) -> (Option<String>, Option<String>) {
    if let Some(rest) = model.strip_prefix("OVERRIDE-MODEL-") {
        return (None, Some(rest.to_string()));
    }
    if let Some(rest) = model.strip_prefix("OVERRIDE-") {
        return (Some(rest.to_string()), None);
    }
    (None, None)
}

// OpenAI Middleware Logic
fn apply_openai_system_prompt_patch(
    req: &mut OpenAIChatCompletionRequest,
    rules: &[SystemPromptRule],
) {
    if rules.is_empty() {
        return;
    }

    // 1. Consolidate current system prompt
    let mut current_system = String::new();

    // We only capture the text content. We will overwrite system messages later.
    for msg in &req.messages {
        if msg.role == "system" {
            if let Some(content) = &msg.content {
                match content {
                    OpenAIContent::String(s) => {
                        if !current_system.is_empty() {
                            current_system.push_str("\n\n");
                        }
                        current_system.push_str(s);
                    }
                    OpenAIContent::Array(parts) => {
                        for part in parts {
                            if let OpenAIContentPart::Text { text } = part {
                                if !current_system.is_empty() {
                                    current_system.push_str("\n\n");
                                }
                                current_system.push_str(text);
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Find matching rules
    for rule in rules {
        let mut all_matched = true;
        for pattern in &rule.r#match {
            if pattern == "ALL" {
                continue; // Matches everything
            }
            if let Ok(re) = Regex::new(pattern) {
                if !re.is_match(&current_system) {
                    all_matched = false;
                    break;
                }
            } else if let Ok(re) = glob_to_regex(pattern) {
                if !re.is_match(&current_system) {
                    all_matched = false;
                    break;
                }
            } else if !current_system.contains(pattern) {
                all_matched = false;
                break;
            }
        }

        if all_matched {
            // Apply Actions sequentially
            for action in &rule.actions {
                match action {
                    SystemPromptOp::Replace { pattern, with } => {
                        if let Ok(re) = Regex::new(pattern) {
                            current_system = re.replace_all(&current_system, with).to_string();
                        }
                    }
                    SystemPromptOp::Prepend { value } => {
                        if !current_system.is_empty() {
                            current_system = format!("{}\n\n{}", value, current_system);
                        } else {
                            current_system = value.clone();
                        }
                    }
                    SystemPromptOp::Append { value } => {
                        if !current_system.is_empty() {
                            current_system = format!("{}\n\n{}", current_system, value);
                        } else {
                            current_system = value.clone();
                        }
                    }
                    SystemPromptOp::MoveToUser {
                        forced_system_prompt,
                        prefix,
                        suffix,
                    } => {
                        // Prepare the content to move
                        let mut moved_content = current_system.clone();
                        if let Some(p) = prefix {
                            moved_content = format!("{}{}", p, moved_content);
                        }
                        if let Some(s) = suffix {
                            moved_content = format!("{}{}", moved_content, s);
                        }

                        // Update system prompt (clear or force)
                        if let Some(forced) = forced_system_prompt {
                            current_system = forced.clone();
                        } else {
                            current_system = String::new();
                        }

                        // Inject into User message
                        inject_into_openai_first_user_message(req, &moved_content);
                    }
                    SystemPromptOp::Delete => {
                        current_system.clear();
                    }
                }
            }
        }
    }

    // 3. Write back system prompt to request
    update_openai_system_content(req, current_system);
}

fn update_openai_system_content(req: &mut OpenAIChatCompletionRequest, new_content: String) {
    if new_content.is_empty() {
        // Remove all system messages
        req.messages.retain(|m| m.role != "system");
        return;
    }

    // Check if we have system messages
    let has_system = req.messages.iter().any(|m| m.role == "system");

    if has_system {
        // Replace first, remove others
        let mut first_set = false;
        req.messages.retain_mut(|m| {
            if m.role == "system" {
                if !first_set {
                    m.content = Some(OpenAIContent::String(new_content.clone()));
                    first_set = true;
                    true
                } else {
                    false // Remove subsequent
                }
            } else {
                true
            }
        });
    } else {
        // Insert at top
        req.messages.insert(
            0,
            OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIContent::String(new_content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
            },
        );
    }
}

fn inject_into_openai_first_user_message(req: &mut OpenAIChatCompletionRequest, content: &str) {
    if content.is_empty() {
        return;
    }

    if let Some(first_user) = req.messages.iter_mut().find(|m| m.role == "user") {
        match &mut first_user.content {
            Some(OpenAIContent::String(s)) => {
                *s = format!("{}\n\n{}", content, s);
            }
            Some(OpenAIContent::Array(parts)) => {
                parts.insert(
                    0,
                    OpenAIContentPart::Text {
                        text: content.to_string(),
                    },
                );
            }
            None => {
                first_user.content = Some(OpenAIContent::String(content.to_string()));
            }
        }
    } else {
        // Insert new user message
        req.messages.insert(
            0,
            OpenAIMessage {
                role: "user".to_string(),
                content: Some(OpenAIContent::String(content.to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
            },
        );
    }
}

fn apply_openai_tool_filter(
    req: &mut OpenAIChatCompletionRequest,
    config: &Option<ToolFilterConfig>,
) {
    let config = match config {
        Some(c) => c,
        None => return,
    };

    if let Some(tools) = &mut req.tools {
        tools.retain(|tool| {
            let name = &tool.function.name;

            // Deny logic
            if let Some(deny_list) = &config.deny {
                for pattern in deny_list {
                    if let Ok(re) = Regex::new(pattern) {
                        if re.is_match(name) {
                            return false;
                        }
                    } else if let Ok(re) = glob_to_regex(pattern) {
                        if re.is_match(name) {
                            return false;
                        }
                    }
                }
            }

            // Allow logic
            if let Some(allow_list) = &config.allow {
                let mut matched = false;
                for pattern in allow_list {
                    if let Ok(re) = Regex::new(pattern) {
                        if re.is_match(name) {
                            matched = true;
                            break;
                        }
                    } else if let Ok(re) = glob_to_regex(pattern) {
                        if re.is_match(name) {
                            matched = true;
                            break;
                        }
                    }
                }
                if !matched {
                    return false;
                }
            }
            true
        });
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
    use bytes::Bytes;
    use futures::stream;

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

    #[test]
    fn test_check_overrides() {
        assert_eq!(check_overrides("gpt-4"), (None, None));
        assert_eq!(
            check_overrides("OVERRIDE-myprofile"),
            (Some("myprofile".to_string()), None)
        );
        assert_eq!(
            check_overrides("OVERRIDE-MODEL-claude-3"),
            (None, Some("claude-3".to_string()))
        );
        assert_eq!(
            check_overrides("OVERRIDE-MODEL-something"),
            (None, Some("something".to_string()))
        );
    }
}
