use crate::protocol::*;
use anyhow::Result;
use serde_json::{json, Value};

pub fn convert_request(
    req: AnthropicMessageRequest,
    resolved_model: String,
) -> Result<OpenAIChatCompletionRequest> {
    let mut openai_messages = Vec::new();

    // 1. Handle System Messages
    if let Some(system) = req.system {
        match system {
            SystemPrompt::String(s) => {
                openai_messages.push(OpenAIMessage {
                    role: "system".to_string(),
                    content: Some(OpenAIContent::String(s)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            SystemPrompt::Array(blocks) => {
                for block in blocks {
                    if block.r#type == "text" {
                        openai_messages.push(OpenAIMessage {
                            role: "system".to_string(),
                            content: Some(OpenAIContent::String(block.text)),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
            }
        }
    }

    // 2. Process Messages (Transformation & Tool Result Processing)
    for msg in req.messages {
        match msg.content {
            AnthropicMessageContent::String(s) => {
                openai_messages.push(OpenAIMessage {
                    role: msg.role,
                    content: Some(OpenAIContent::String(s)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            AnthropicMessageContent::Blocks(blocks) => {
                // Separate blocks into tool_results and others (text/image/tool_use)
                let mut tool_results = Vec::new();
                let mut other_blocks = Vec::new();

                for block in blocks {
                    match block {
                        AnthropicContentBlock::ToolResult { .. } => tool_results.push(block),
                        _ => other_blocks.push(block),
                    }
                }

                // 2a. Process Tool Results FIRST
                for tr in tool_results {
                    if let AnthropicContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error: _,
                    } = tr
                    {
                        let content_str = match content {
                            Some(c) => normalize_content_to_string(c).unwrap_or_default(),
                            None => "".to_string(),
                        };

                        openai_messages.push(OpenAIMessage {
                            role: "tool".to_string(),
                            content: Some(OpenAIContent::String(content_str)),
                            name: None,
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id),
                        });
                    }
                }

                // 2b. Process Remaining Content
                if !other_blocks.is_empty() {
                    let mut content_parts = Vec::new();
                    let mut tool_calls = Vec::new();

                    for block in other_blocks {
                        match block {
                            AnthropicContentBlock::Text { text } => {
                                content_parts.push(OpenAIContentPart::Text { text });
                            }
                            AnthropicContentBlock::Image { source } => {
                                // Convert Anthropic image to OpenAI image_url
                                let image_url = match source {
                                    AnthropicImageSource::Base64 { media_type, data } => {
                                        format!("data:{};base64,{}", media_type, data)
                                    }
                                    AnthropicImageSource::Url { url } => url,
                                };

                                content_parts.push(OpenAIContentPart::ImageUrl {
                                    image_url: OpenAIImageUrl {
                                        url: image_url,
                                        detail: Some("auto".to_string()),
                                    },
                                });
                            }
                            AnthropicContentBlock::ToolUse { id, name, input } => {
                                tool_calls.push(OpenAIToolCall {
                                    id,
                                    r#type: "function".to_string(),
                                    function: OpenAIFunctionCall {
                                        name,
                                        arguments: serde_json::to_string(&input)
                                            .unwrap_or_default(),
                                    },
                                });
                            }
                            AnthropicContentBlock::Thinking {
                                thinking,
                                signature: _,
                            } => {
                                content_parts.push(OpenAIContentPart::Text {
                                    text: format!("<thinking>{}</thinking>", thinking),
                                });
                            }
                            _ => {}
                        }
                    }

                    // Determine final content structure (String vs Array)
                    let final_content = if content_parts.is_empty() {
                        None
                    } else if content_parts.len() == 1 {
                        // Optimization: if just one text part, verify if we can send as string
                        match &content_parts[0] {
                            OpenAIContentPart::Text { text } => {
                                Some(OpenAIContent::String(text.clone()))
                            }
                            _ => Some(OpenAIContent::Array(content_parts)),
                        }
                    } else {
                        Some(OpenAIContent::Array(content_parts))
                    };

                    let final_tool_calls = if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    };

                    if final_content.is_some() || final_tool_calls.is_some() {
                        openai_messages.push(OpenAIMessage {
                            role: msg.role.clone(),
                            content: final_content,
                            name: None,
                            tool_calls: final_tool_calls,
                            tool_call_id: None,
                        });
                    }
                }
            }
        }
    }

    // 3. Handle Tools
    let mut openai_tools = None;
    if let Some(tools) = req.tools {
        let mut converted_tools = Vec::new();
        for tool in tools {
            if tool.name == "BatchTool" {
                continue;
            }

            let mut schema = tool.input_schema;
            remove_uri_format(&mut schema);

            converted_tools.push(OpenAITool {
                r#type: "function".to_string(),
                function: OpenAIFunction {
                    name: tool.name,
                    description: tool.description,
                    parameters: schema,
                },
            });
        }
        if !converted_tools.is_empty() {
            openai_tools = Some(converted_tools);
        }
    }

    // 4. Handle Tool Choice
    let tool_choice = req.tool_choice.map(|tc| match tc {
        AnthropicToolChoice::Auto => json!("auto"),
        AnthropicToolChoice::Any => json!("required"),
        AnthropicToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
    });

    Ok(OpenAIChatCompletionRequest {
        model: resolved_model,
        messages: openai_messages,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        stop: req.stop_sequences.map(Value::from),
        max_tokens: req.max_tokens,
        tools: openai_tools,
        tool_choice,
        presence_penalty: None,
        frequency_penalty: None,
        user: None,
    })
}

// Helper: Normalize to string for Tool Result content (which is typically just text/json)
fn normalize_content_to_string(content: AnthropicMessageContent) -> Option<String> {
    match content {
        AnthropicMessageContent::String(s) => Some(s),
        AnthropicMessageContent::Blocks(blocks) => {
            let texts: Vec<String> = blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.clone()),
                    _ => None, // Tool results usually don't have images
                })
                .collect();

            if texts.is_empty() {
                None
            } else {
                Some(texts.join(" "))
            }
        }
    }
}

fn remove_uri_format(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            let is_string = map.get("type").and_then(|v| v.as_str()) == Some("string");
            let is_uri = map.get("format").and_then(|v| v.as_str()) == Some("uri");

            if is_string && is_uri {
                map.remove("format");
            }

            for (_, v) in map.iter_mut() {
                remove_uri_format(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                remove_uri_format(v);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_uri_format() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri"
                },
                "nested": {
                    "type": "object",
                    "properties": {
                        "site": {
                            "type": "string",
                            "format": "uri"
                        }
                    }
                }
            }
        });

        remove_uri_format(&mut schema);

        let url_format = schema["properties"]["url"].get("format");
        assert!(url_format.is_none());

        let nested_format = schema["properties"]["nested"]["properties"]["site"].get("format");
        assert!(nested_format.is_none());
    }
}
