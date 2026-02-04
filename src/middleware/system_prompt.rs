use super::Middleware;
use crate::config::{glob_to_regex, SystemPromptOp, SystemPromptRule};
use crate::protocol::{
    AnthropicContentBlock, AnthropicMessage, AnthropicMessageContent, AnthropicMessageRequest,
    SystemPrompt,
};
use anyhow::Result;
use regex::Regex;

pub struct SystemPromptPatcherMiddleware {
    pub rules: Vec<SystemPromptRule>,
}

impl SystemPromptPatcherMiddleware {
    pub fn new(rules: Vec<SystemPromptRule>) -> Self {
        Self { rules }
    }
}

impl Middleware for SystemPromptPatcherMiddleware {
    fn on_request(&self, req: &mut AnthropicMessageRequest) -> Result<()> {
        if self.rules.is_empty() {
            return Ok(());
        }

        // 1. Consolidate system prompt to string for matching and modification
        // We will work on a mutable String buffer.
        let mut current_system = match &req.system {
            Some(SystemPrompt::String(s)) => s.clone(),
            Some(SystemPrompt::Array(blocks)) => blocks
                .iter()
                .map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n\n"),
            None => String::new(),
        };

        // 2. Find matching rules
        for rule in &self.rules {
            // If system prompt is empty (e.g. was moved/deleted), we might still match if pattern allows it?
            // "if ALL strings/regexps/globs in match are matched". Empty string might not match much.

            let mut all_matched = true;
            for pattern in &rule.r#match {
                if pattern == "ALL" {
                    continue; // Matches everything
                }

                // Try as raw regex first (partial match allowed)
                if let Ok(re) = Regex::new(pattern) {
                    if !re.is_match(&current_system) {
                        all_matched = false;
                        break;
                    }
                } else if let Ok(re) = glob_to_regex(pattern) {
                    // Fallback to glob (full match)
                    if !re.is_match(&current_system) {
                        all_matched = false;
                        break;
                    }
                } else {
                    // Pattern is neither valid regex nor simple glob. Treat as substring?
                    // Or just fail match.
                    if !current_system.contains(pattern) {
                        all_matched = false;
                        break;
                    }
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

                            // Update system prompt for the request (clear it or set to forced)
                            if let Some(forced) = forced_system_prompt {
                                current_system = forced.clone();
                            } else {
                                current_system = String::new();
                            }

                            // Inject into User message
                            inject_into_first_user_message(req, &moved_content);
                        }
                        SystemPromptOp::Delete => {
                            // Clear the current system prompt buffer
                            current_system.clear();
                        }
                    }
                }
            }
        }

        // Finalize: update req.system with the modified buffer
        if current_system.is_empty() {
            req.system = None;
        } else {
            req.system = Some(SystemPrompt::String(current_system));
        }

        Ok(())
    }
}

fn inject_into_first_user_message(req: &mut AnthropicMessageRequest, content: &str) {
    if content.is_empty() {
        return;
    }

    if let Some(first_msg) = req.messages.iter_mut().find(|m| m.role == "user") {
        match &mut first_msg.content {
            AnthropicMessageContent::String(s) => {
                *s = format!("{}\n\n{}", content, s);
            }
            AnthropicMessageContent::Blocks(blocks) => {
                // Prepend a text block
                blocks.insert(
                    0,
                    AnthropicContentBlock::Text {
                        text: content.to_string(),
                    },
                );
            }
        }
    } else {
        // No user message found. Insert a new one at the beginning.
        req.messages.insert(
            0,
            AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::String(content.to_string()),
            },
        );
    }
}
