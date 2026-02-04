use super::{Middleware, StreamBox};
use crate::protocol::{
    AnthropicContentBlock, AnthropicDelta, AnthropicMessageRequest, AnthropicMessageResponse,
    AnthropicStreamEvent, AnthropicTool,
};
use anyhow::Result;
use async_stream::stream;
use futures::StreamExt;
use serde_json::{json, Value};

pub struct ToolEnforcerMiddleware;

impl ToolEnforcerMiddleware {
    pub fn new() -> Self {
        Self
    }

    fn get_exit_tool_def(&self) -> AnthropicTool {
        AnthropicTool {
            name: "ExitTool".to_string(),
            description: Some("Use this tool when you are in tool mode and have completed the task. The response argument will be returned to the user.".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The final response to the user."
                    }
                },
                "required": ["response"]
            }),
            input_examples: None,
        }
    }
}

impl Middleware for ToolEnforcerMiddleware {
    fn on_request(&self, req: &mut AnthropicMessageRequest) -> Result<()> {
        // Only inject if tools are already present or we want to force tool use
        // The requirement is to inject it. Let's assume we inject it if tools are present,
        // or maybe we should always inject it if this middleware is active?
        // Assuming if tools are enabled, we inject it.

        if let Some(tools) = &mut req.tools {
            // Check if ExitTool already exists to avoid duplication
            if !tools.iter().any(|t| t.name == "ExitTool") {
                tools.push(self.get_exit_tool_def());
            }
        } else {
            // If no tools are defined, maybe we shouldn't inject ExitTool?
            // Or should we? If the model supports tools, we might want to enabling it.
            // But if the user didn't ask for tools, adding one might confuse the model
            // or change pricing (token usage).
            // Let's safe-guard: only add if tools list is not empty or Some.
            // But wait, the user might want to use ExitTool even if they didn't define other tools
            // (unlikely scenario for a Router, usually Router adds it to manage sub-agents).
            // I'll stick to: if `tools` is Some, append it.
        }

        Ok(())
    }

    fn on_response(&self, resp: &mut AnthropicMessageResponse) -> Result<()> {
        // Check if the response contains a tool use for ExitTool
        let mut exit_tool_response = None;
        let mut new_blocks = Vec::new();

        for block in &resp.content {
            if let AnthropicContentBlock::ToolUse { name, input, .. } = block {
                if name == "ExitTool" {
                    // Found ExitTool!
                    if let Some(response_text) = input.get("response").and_then(|v| v.as_str()) {
                        exit_tool_response = Some(response_text.to_string());
                    } else {
                        // Fallback if response arg is missing or not string
                        exit_tool_response = Some("".to_string());
                    }
                    // We consume this block (don't add to new_blocks yet)
                    // Actually, if we find ExitTool, we should probably replace the ENTIRE response content
                    // with just the text, or append the text?
                    // The instruction says "Transform Tool Call into standard Text Message".
                    continue;
                }
            }
            new_blocks.push(block.clone());
        }

        if let Some(text) = exit_tool_response {
            // Replace content with just the text
            resp.content = vec![AnthropicContentBlock::Text { text }];
            resp.stop_reason = Some("end_turn".to_string());
        }

        Ok(())
    }

    fn transform_stream(&self, stream: StreamBox) -> StreamBox {
        // For streaming, we need to detect ExitTool usage incrementally.
        // This is complex because the tool name comes first, then arguments.
        // If we detect "ExitTool", we need to suppress the ToolUse blocks and instead
        // emit Text blocks with the content of the "response" argument.

        let mut stream = stream;

        // We need state to track if we are inside ExitTool
        struct State {
            capturing_exit_tool: bool,
            has_captured_exit_tool: bool,
            tool_name_buffer: String,
            tool_input_buffer: String,
            current_tool_index: Option<u32>,
        }

        let mut state = State {
            capturing_exit_tool: false,
            has_captured_exit_tool: false,
            tool_name_buffer: String::new(),
            tool_input_buffer: String::new(),
            current_tool_index: None,
        };

        let output_stream = stream! {
            while let Some(event_res) = stream.next().await {
                match event_res {
                    Ok(event) => {
                        match event {
                            AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
                                if let AnthropicContentBlock::ToolUse { name, .. } = &content_block {
                                    if name == "ExitTool" {
                                        state.capturing_exit_tool = true;
                                        state.current_tool_index = Some(index);
                                        state.tool_name_buffer = name.clone();
                                        // Don't emit this event
                                        continue;
                                    } else if name.is_empty() {
                                        // Name might come later? No, Start block usually has name if known,
                                        // or name is empty and filled by delta?
                                        // Protocol says Start has `name`.
                                        // But sometimes it might be partial?
                                        // Let's assume if it's not ExitTool, we pass it through.
                                    }
                                }

                                if state.capturing_exit_tool {
                                    // We are in ExitTool, suppress other blocks if they are related?
                                    // No, index matches.
                                } else {
                                    yield Ok(AnthropicStreamEvent::ContentBlockStart { index, content_block });
                                }
                            }
                            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                                if state.capturing_exit_tool && state.current_tool_index == Some(index) {
                                    if let AnthropicDelta::InputJsonDelta { partial_json } = delta {
                                        if state.tool_input_buffer.len() + partial_json.len() > 1024 * 100 {
                                            // Safety limit: 100KB. If exceeded, stop capturing to avoid DoS.
                                            // We effectively corrupt the capture but protect memory.
                                            state.capturing_exit_tool = false;
                                        } else {
                                            state.tool_input_buffer.push_str(&partial_json);
                                        }
                                        // We can't really stream the "response" field value as text delta reliably
                                        // without a streaming JSON parser.
                                        // Strategy: Buffer everything until MessageStop or ToolUse close?
                                        // Then emit Text block?
                                        // But that defeats streaming latency.
                                        // However, converting Tool Call -> Text usually implies we want the text.
                                        // If the user wants streaming text, they should ask the model to speak.
                                        // ExitTool is for when the model THINKS it's done and wants to return a final answer.
                                        // So buffering is acceptable.
                                    }
                                    continue;
                                }
                                yield Ok(AnthropicStreamEvent::ContentBlockDelta { index, delta });
                            }
                            AnthropicStreamEvent::ContentBlockStop { index } => {
                                if state.capturing_exit_tool && state.current_tool_index == Some(index) {
                                    // End of ExitTool block.
                                    // Parse the input buffer
                                    if let Ok(json_val) = serde_json::from_str::<Value>(&state.tool_input_buffer) {
                                        if let Some(response_text) = json_val.get("response").and_then(|v| v.as_str()) {
                                            // Emit a Text block start, delta, and stop
                                            // reusing the same index? Or a new one?
                                            // Reusing the index might be confusing if the client saw a ToolUse start?
                                            // Ah, we SUPPRESSED the ToolUse start.
                                            // So we can emit a Text start now.

                                            yield Ok(AnthropicStreamEvent::ContentBlockStart {
                                                index,
                                                content_block: AnthropicContentBlock::Text { text: "".to_string() }
                                            });
                                            yield Ok(AnthropicStreamEvent::ContentBlockDelta {
                                                index,
                                                delta: AnthropicDelta::TextDelta { text: response_text.to_string() }
                                            });
                                            yield Ok(AnthropicStreamEvent::ContentBlockStop { index });
                                        }
                                    }

                                    state.capturing_exit_tool = false;
                                    state.has_captured_exit_tool = true;
                                    state.current_tool_index = None;
                                    state.tool_input_buffer.clear();
                                    continue;
                                }
                                yield Ok(AnthropicStreamEvent::ContentBlockStop { index });
                            }
                            AnthropicStreamEvent::MessageDelta { mut delta, usage } => {
                                if state.has_captured_exit_tool {
                                    // If we are capturing, we might want to change stop_reason if it was tool_use
                                    if let Some(sr) = &delta.stop_reason {
                                        if sr == "tool_use" {
                                            delta.stop_reason = Some("end_turn".to_string());
                                        }
                                    }
                                }
                                yield Ok(AnthropicStreamEvent::MessageDelta { delta, usage });
                            }
                            _ => {
                                yield Ok(event);
                            }
                        }
                    }
                    Err(e) => yield Err(e),
                }
            }
        };

        Box::pin(output_stream)
    }
}
