use crate::config::ModelConfig;
use crate::protocol::*;
use anyhow::Result;
use async_stream::stream;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub fn convert_response(resp: OpenAIChatCompletionResponse, model_config: Option<&ModelConfig>) -> Result<AnthropicMessageResponse> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;
    let msg = &choice.message;

    // 1. ID Transformation
    let id = if resp.id.starts_with("chatcmpl") {
        resp.id.replace("chatcmpl", "msg")
    } else {
        format!("msg_{}", Uuid::new_v4().simple())
    };

    // 2. Content Construction
    let mut content_blocks = Vec::new();

    // Handle Reasoning (Thinking)
    if let Some(r) = &msg.reasoning {
        let r_text = if let Value::String(s) = r {
            s.clone()
        } else {
            r.to_string()
        };
        
        if !r_text.is_empty() {
            content_blocks.push(AnthropicContentBlock::Thinking {
                signature: "signature".to_string(), // Placeholder
                thinking: r_text,
            });
        }
    }

    // Text content
    if let Some(content) = &msg.content {
        match content {
            OpenAIContent::String(text) => {
                if !text.is_empty() {
                    content_blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                }
            }
            OpenAIContent::Array(parts) => {
                for part in parts {
                    // Anthropic response doesn't usually include images back to user in this format,
                    // usually just text/tool_use. If OpenAI echoes images, we might skip or try to map?
                    // For now, only mapping text back.
                    if let OpenAIContentPart::Text { text } = part {
                        content_blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                    }
                }
            }
        }
    }

    // Tool calls
    if let Some(tool_calls) = &msg.tool_calls {
        let json_repair = model_config.and_then(|c| c.preprocess.as_ref()).and_then(|p| p.json_repair).unwrap_or(false);
        for tc in tool_calls {
            let input_val: Value = parse_tool_arguments(&tc.function.arguments, json_repair);

            content_blocks.push(AnthropicContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input: input_val,
            });
        }
    }

    // 3. Stop Reason Mapping
    let stop_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => Some("tool_use".to_string()),
        Some("stop") => Some("end_turn".to_string()),
        Some("length") => Some("max_tokens".to_string()),
        _ => Some("end_turn".to_string()),
    };

    // 4. Usage
    let usage = if let Some(u) = resp.usage {
        AnthropicUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        }
    } else {
        // Fallback estimation
        let input_tokens = estimate_tokens_content(&msg.content);
        let output_tokens = estimate_tokens_content(&msg.content); // Rough approx
        AnthropicUsage {
            input_tokens,
            output_tokens,
        }
    };

    Ok(AnthropicMessageResponse {
        id,
        r#type: "message".to_string(),
        role: "assistant".to_string(),
        content: content_blocks,
        model: resp.model,
        stop_reason,
        stop_sequence: None,
        usage,
    })
}

fn estimate_tokens_content(content: &Option<OpenAIContent>) -> u32 {
    match content {
        Some(OpenAIContent::String(s)) => estimate_tokens(s),
        Some(OpenAIContent::Array(parts)) => parts
            .iter()
            .map(|p| match p {
                OpenAIContentPart::Text { text } => estimate_tokens(text),
                _ => 0,
            })
            .sum(),
        None => 0,
    }
}

fn estimate_tokens(text: &str) -> u32 {
    text.split_whitespace().count() as u32
}

fn parse_tool_arguments(args: &str, use_repair: bool) -> Value {
    // 1. Try standard JSON parsing first (fast path)
    if let Ok(v) = serde_json::from_str(args) {
        return v;
    }

    // 2. Try json5 parsing if enabled (handles trailing commas, unquoted keys, single quotes)
    if use_repair {
        if let Ok(v) = json5::from_str(args) {
            return v;
        }
    }

    // 3. Fallback: Return empty object if parsing fails completely
    json!({})
}

pub fn record_stream<S, F>(
    input_stream: S,
    callback: F,
) -> impl Stream<Item = Result<AnthropicStreamEvent, anyhow::Error>>
where
    S: Stream<Item = Result<AnthropicStreamEvent, anyhow::Error>> + Unpin,
    F: FnOnce(AnthropicMessageResponse) + Send + 'static,
{
    let mut input_stream = input_stream;
    let mut accumulated_response: Option<AnthropicMessageResponse> = None;
    let mut callback_opt = Some(callback);
    let mut partial_tool_inputs: HashMap<u32, String> = HashMap::new();

    stream! {
        while let Some(event_res) = input_stream.next().await {
            if let Ok(event) = &event_res {
                match event {
                    AnthropicStreamEvent::MessageStart { message } => {
                        accumulated_response = Some(message.clone());
                    }
                    AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
                        if let Some(resp) = &mut accumulated_response {
                            // Ensure content vector is large enough
                            while resp.content.len() <= *index as usize {
                                // Add a dummy block that will be overwritten
                                resp.content.push(AnthropicContentBlock::Text { text: "".to_string() });
                            }
                            resp.content[*index as usize] = content_block.clone();
                        }
                    }
                    AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                        if let Some(resp) = &mut accumulated_response {
                            if let Some(block) = resp.content.get_mut(*index as usize) {
                                match (block, delta) {
                                    (AnthropicContentBlock::Text { text }, AnthropicDelta::TextDelta { text: delta_text }) => {
                                        text.push_str(delta_text);
                                    }
                                    (AnthropicContentBlock::Thinking { thinking, .. }, AnthropicDelta::ThinkingDelta { thinking: delta_thinking }) => {
                                        thinking.push_str(delta_thinking);
                                    }
                                    (AnthropicContentBlock::ToolUse { .. }, AnthropicDelta::InputJsonDelta { partial_json }) => {
                                        let json_buf = partial_tool_inputs.entry(*index).or_default();
                                        json_buf.push_str(partial_json);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    AnthropicStreamEvent::MessageDelta { delta, usage } => {
                        if let Some(resp) = &mut accumulated_response {
                            if let Some(sr) = &delta.stop_reason {
                                resp.stop_reason = Some(sr.clone());
                            }
                            if let Some(ss) = &delta.stop_sequence {
                                resp.stop_sequence = Some(ss.clone());
                            }
                            if usage.input_tokens > 0 {
                                resp.usage.input_tokens = usage.input_tokens;
                            }
                            if usage.output_tokens > 0 {
                                resp.usage.output_tokens = usage.output_tokens;
                            }
                        }
                    }
                    AnthropicStreamEvent::MessageStop => {
                        if let Some(mut resp) = accumulated_response.take() {
                            // Finalize tool inputs
                            for (index, json_str) in partial_tool_inputs.drain() {
                                if let Some(AnthropicContentBlock::ToolUse { input, .. }) = resp.content.get_mut(index as usize) {
                                    if let Ok(val) = serde_json::from_str(json_str.as_str()) {
                                        *input = val;
                                    }
                                }
                            }

                            if let Some(cb) = callback_opt.take() {
                                cb(resp);
                            }
                        }
                    }
                    _ => {}
                }
            }
            yield event_res;
        }
    }
}

// Streaming State
#[derive(Debug, PartialEq, Clone, Copy)]
enum BlockType {
    Text,
    Thinking,
    ToolUse,
}

struct StreamState {
    block_index: u32,
    active_block_type: Option<BlockType>,
    open_block_indices: HashSet<u32>,
    tool_index_map: HashMap<u32, u32>, // OpenAI index -> Anthropic block index
    msg_id: String,
    model: String,
}

pub fn convert_stream<S>(input_stream: S) -> impl Stream<Item = Result<AnthropicStreamEvent>>
where
    S: Stream<Item = Result<OpenAIChatCompletionChunk, anyhow::Error>>,
{
    stream! {
        let mut input_stream = Box::pin(input_stream);
        let mut state = StreamState {
            block_index: 0,
            active_block_type: None,
            open_block_indices: HashSet::new(),
            tool_index_map: HashMap::new(),
            msg_id: format!("msg_{}", Uuid::new_v4().simple()),
            model: "".to_string(),
        };

        let mut first_chunk = true;

        while let Some(chunk_result) = input_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    // Initialize on first chunk
                    if first_chunk {
                        state.model = chunk.model.clone();
                        if !chunk.id.is_empty() {
                            state.msg_id = if chunk.id.starts_with("chatcmpl") {
                                chunk.id.replace("chatcmpl", "msg")
                            } else {
                                chunk.id
                            };
                        }

                        // Send message_start
                        yield Ok(AnthropicStreamEvent::MessageStart {
                            message: AnthropicMessageResponse {
                                id: state.msg_id.clone(),
                                r#type: "message".to_string(),
                                role: "assistant".to_string(),
                                content: vec![],
                                model: state.model.clone(),
                                stop_reason: None,
                                stop_sequence: None,
                                usage: AnthropicUsage { input_tokens: 0, output_tokens: 0 },
                            },
                        });

                        // Send ping
                        yield Ok(AnthropicStreamEvent::Ping);

                        first_chunk = false;
                    }

                    if let Some(choice) = chunk.choices.first() {
                        let delta = &choice.delta;

                        // 1. Handle Reasoning (Thinking)
                        let reasoning = delta.reasoning.as_deref().or(delta.reasoning_content.as_deref());
                        if let Some(r_text) = reasoning {
                            if !r_text.is_empty() {
                                if state.active_block_type != Some(BlockType::Thinking) {
                                    // Close previous blocks
                                    for idx in state.open_block_indices.drain() {
                                        yield Ok(AnthropicStreamEvent::ContentBlockStop { index: idx });
                                    }

                                    // Increment index if we had previous blocks
                                    if state.active_block_type.is_some() {
                                        state.block_index += 1;
                                    }

                                    state.active_block_type = Some(BlockType::Thinking);
                                    state.open_block_indices.insert(state.block_index);

                                    // Start thinking block
                                    yield Ok(AnthropicStreamEvent::ContentBlockStart {
                                        index: state.block_index,
                                        content_block: AnthropicContentBlock::Thinking {
                                            signature: "signature".to_string(), // placeholder
                                            thinking: "".to_string(),
                                        },
                                    });
                                }

                                yield Ok(AnthropicStreamEvent::ContentBlockDelta {
                                    index: state.block_index,
                                    delta: AnthropicDelta::ThinkingDelta {
                                        thinking: r_text.to_string(),
                                    },
                                });
                            }
                        }

                        // 2. Handle Text Content
                        if let Some(content) = &delta.content {
                            if !content.is_empty() {
                                if state.active_block_type != Some(BlockType::Text) {
                                    // Close previous blocks
                                    for idx in state.open_block_indices.drain() {
                                        yield Ok(AnthropicStreamEvent::ContentBlockStop { index: idx });
                                    }

                                    // Increment index if we had previous blocks
                                    if state.active_block_type.is_some() {
                                        state.block_index += 1;
                                    }

                                    state.active_block_type = Some(BlockType::Text);
                                    state.open_block_indices.insert(state.block_index);

                                    // Start text block
                                    yield Ok(AnthropicStreamEvent::ContentBlockStart {
                                        index: state.block_index,
                                        content_block: AnthropicContentBlock::Text {
                                            text: "".to_string(),
                                        },
                                    });
                                }

                                yield Ok(AnthropicStreamEvent::ContentBlockDelta {
                                    index: state.block_index,
                                    delta: AnthropicDelta::TextDelta {
                                        text: content.clone(),
                                    },
                                });
                            }
                        }

                        // 3. Handle Tool Calls
                        if let Some(tool_calls) = &delta.tool_calls {
                            if state.active_block_type != Some(BlockType::ToolUse) {
                                // If we were doing something else, close it
                                if state.active_block_type.is_some() {
                                    for idx in state.open_block_indices.drain() {
                                        yield Ok(AnthropicStreamEvent::ContentBlockStop { index: idx });
                                    }
                                    state.block_index += 1;
                                }
                                state.active_block_type = Some(BlockType::ToolUse);
                            }

                            for tc in tool_calls {
                                // If it has an ID, it's a new tool call block
                                if let Some(id) = &tc.id {
                                    // New tool call, map its OpenAI index to our current sequential block index
                                    state.tool_index_map.insert(tc.index, state.block_index);

                                    state.open_block_indices.insert(state.block_index);

                                    yield Ok(AnthropicStreamEvent::ContentBlockStart {
                                        index: state.block_index,
                                        content_block: AnthropicContentBlock::ToolUse {
                                            id: id.clone(),
                                            name: tc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default(),
                                            input: json!({}), // Empty object as placeholder
                                        },
                                    });

                                    // Prepare for next potential tool call (indices must be sequential in Anthropic)
                                    state.block_index += 1;
                                }

                                // If it has arguments, send delta to the mapped block index
                                if let Some(func) = &tc.function {
                                    if let Some(args) = &func.arguments {
                                        if let Some(anthropic_idx) = state.tool_index_map.get(&tc.index) {
                                            yield Ok(AnthropicStreamEvent::ContentBlockDelta {
                                                index: *anthropic_idx,
                                                delta: AnthropicDelta::InputJsonDelta {
                                                    partial_json: args.clone(),
                                                },
                                            });
                                        }
                                    }
                                }
                            }
                        }

                        // 4. Handle Finish Reason
                        if let Some(finish) = &choice.finish_reason {
                            // Close any open blocks
                            for idx in state.open_block_indices.drain() {
                                yield Ok(AnthropicStreamEvent::ContentBlockStop { index: idx });
                            }

                            let stop_reason = match finish.as_str() {
                                "tool_calls" => "tool_use",
                                "stop" => "end_turn",
                                "length" => "max_tokens",
                                _ => "end_turn",
                            };

                            yield Ok(AnthropicStreamEvent::MessageDelta {
                                delta: AnthropicMessageDelta {
                                    stop_reason: Some(stop_reason.to_string()),
                                    stop_sequence: None,
                                },
                                usage: AnthropicUsage { input_tokens: 0, output_tokens: 0 }, // Usage might come in message_stop or earlier
                            });
                        }
                    }

                    // Handle Usage in last chunk (some providers do this)
                    if let Some(usage) = &chunk.usage {
                         yield Ok(AnthropicStreamEvent::MessageDelta {
                            delta: AnthropicMessageDelta {
                                stop_reason: None,
                                stop_sequence: None,
                            },
                            usage: AnthropicUsage {
                                input_tokens: usage.prompt_tokens,
                                output_tokens: usage.completion_tokens,
                            },
                        });
                    }
                }
                Err(e) => {
                    yield Ok(AnthropicStreamEvent::Error {
                         error: json!({"message": e.to_string()})
                    });
                }
            }
        }

        yield Ok(AnthropicStreamEvent::MessageStop);
    }
}
