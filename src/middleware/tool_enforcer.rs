use super::{Middleware, StreamBox};
use crate::protocol::{
    AnthropicContentBlock, AnthropicDelta, AnthropicMessageRequest, AnthropicMessageResponse,
    AnthropicStreamEvent, AnthropicTool, AnthropicToolDef, AnthropicToolChoice, SystemPrompt,
};
use anyhow::Result;
use async_stream::stream;
use futures::StreamExt;
use serde_json::{json, Value};

pub struct ToolEnforcerMiddleware;

impl Default for ToolEnforcerMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolEnforcerMiddleware {
    pub fn new() -> Self {
        Self
    }

    fn get_exit_tool_def(&self) -> AnthropicTool {
        AnthropicTool::Anthropic(AnthropicToolDef {
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
                "required": ["response"],
                "additionalProperties": false
            }),
            input_examples: None,
            strict: Some(true),
        })
    }
}

impl Middleware for ToolEnforcerMiddleware {
    fn on_request(&self, req: &mut AnthropicMessageRequest) -> Result<()> {
        // Only inject if tools are already present
        if let Some(tools) = &mut req.tools {
            // Check if ExitTool already exists to avoid duplication
            let exists = tools.iter().any(|t| match t {
                AnthropicTool::Anthropic(t) => t.name == "ExitTool",
                AnthropicTool::OpenAI(t) => t.function.name == "ExitTool",
            });

            if !exists {
                tools.push(self.get_exit_tool_def());
            }

            // Force tool choice if not already specific
            if req.tool_choice.is_none() || matches!(req.tool_choice, Some(AnthropicToolChoice::Auto)) {
                 req.tool_choice = Some(AnthropicToolChoice::Any);
            }

            // Inject System Prompt
            let system_prompt = "<system-reminder>Tool mode is active. If no available tool is appropriate, you MUST call the ExitTool.</system-reminder>".to_string();

            match &mut req.system {
                Some(SystemPrompt::String(s)) => {
                    if !s.contains("<system-reminder>") {
                        s.push_str("\n\n");
                        s.push_str(&system_prompt);
                    }
                },
                Some(SystemPrompt::Array(blocks)) => {
                    // Just append a text block
                    blocks.push(crate::protocol::SystemBlock {
                        r#type: "text".to_string(),
                        text: system_prompt,
                        other: std::collections::HashMap::new(),
                    });
                },
                None => {
                    req.system = Some(SystemPrompt::String(system_prompt));
                }
            }
        }

        Ok(())
    }

    fn on_response(&self, resp: &mut AnthropicMessageResponse) -> Result<()> {
        // Check if the response contains a tool use for ExitTool
        let mut new_blocks = Vec::new();

        for block in &resp.content {
            if let AnthropicContentBlock::ToolUse { name, input, .. } = block {
                if name == "ExitTool" {
                    // Found ExitTool!
                    let exit_response_text = if let Some(response_text) = input.get("response").and_then(|v| v.as_str()) {
                        response_text.to_string()
                    } else {
                        "".to_string()
                    };

                    // Replace the ToolUse block with the Text block
                    new_blocks.push(AnthropicContentBlock::Text {
                        text: exit_response_text
                    });

                    // We found ExitTool, so we change stop_reason
                    resp.stop_reason = Some("end_turn".to_string());
                    continue;
                }
            }
            new_blocks.push(block.clone());
        }

        resp.content = new_blocks;

        Ok(())
    }

    fn transform_stream(&self, stream: StreamBox) -> StreamBox {
        let mut stream = stream;

        // We need state to track if we are inside ExitTool
        struct State {
            capturing_exit_tool: bool,
            has_captured_exit_tool: bool,
            tool_input_buffer: String,
            current_tool_index: Option<u32>,
        }

        let mut state = State {
            capturing_exit_tool: false,
            has_captured_exit_tool: false,
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
                                        // Don't emit this event
                                        continue;
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
                                            // Ideally we should maybe error out?
                                            // For now, we just stop buffering but we remain in "capturing" state
                                            // to suppress the rest of the stream for this tool.
                                            // However, without buffering, we can't emit the final text.
                                            // So we will emit an error text or partial?
                                            // Let's just truncate.
                                        } else {
                                            state.tool_input_buffer.push_str(&partial_json);
                                        }
                                    }
                                    continue;
                                }
                                yield Ok(AnthropicStreamEvent::ContentBlockDelta { index, delta });
                            }
                            AnthropicStreamEvent::ContentBlockStop { index } => {
                                if state.capturing_exit_tool && state.current_tool_index == Some(index) {
                                    // End of ExitTool block.
                                    // Parse the input buffer
                                    let response_text = match serde_json::from_str::<Value>(&state.tool_input_buffer) {
                                        Ok(json_val) => {
                                            json_val.get("response")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string())
                                                .unwrap_or_else(|| state.tool_input_buffer.clone())
                                        },
                                        Err(_) => {
                                            // Fallback if JSON is invalid
                                            state.tool_input_buffer.clone()
                                        }
                                    };

                                    // Emit a Text block start, delta, and stop
                                    yield Ok(AnthropicStreamEvent::ContentBlockStart {
                                        index,
                                        content_block: AnthropicContentBlock::Text { text: "".to_string() }
                                    });
                                    yield Ok(AnthropicStreamEvent::ContentBlockDelta {
                                        index,
                                        delta: AnthropicDelta::TextDelta { text: response_text }
                                    });
                                    yield Ok(AnthropicStreamEvent::ContentBlockStop { index });

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
