use super::Middleware;
use crate::config::{SystemPromptAction, SystemPromptRule, glob_to_regex};
use crate::protocol::{AnthropicMessageRequest, SystemPrompt, AnthropicMessage, AnthropicMessageContent, AnthropicContentBlock};
use anyhow::Result;

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

        // 1. Consolidate system prompt to string for matching
        let current_system = match &req.system {
            Some(SystemPrompt::String(s)) => s.clone(),
            Some(SystemPrompt::Array(blocks)) => {
                blocks.iter().map(|b| b.text.clone()).collect::<Vec<_>>().join("\n\n")
            },
            None => String::new(),
        };

        if current_system.is_empty() {
            return Ok(());
        }

        // 2. Find matching rules
        for rule in &self.rules {
            let mut all_matched = true;
            for pattern in &rule.r#match {
                // Try as raw regex first (partial match allowed)
                if let Ok(re) = regex::Regex::new(pattern) {
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
                     all_matched = false; // Invalid regex
                }
            }

            if all_matched {
                // Apply Action
                apply_action(req, &rule.action, &current_system);
                // Assumption: Apply only first matching rule? Or all?
                // "allow general per-systemprompt-name-based transforms"
                // Usually first match or specific strategy. Let's assume sequential application might be chaotic.
                // Stopping after first match is safer for replacements.
                break;
            }
        }

        Ok(())
    }
}

fn apply_action(req: &mut AnthropicMessageRequest, action: &SystemPromptAction, current_content: &str) {
    match action {
        SystemPromptAction::Replace(new_content) => {
            req.system = Some(SystemPrompt::String(new_content.clone()));
        },
        SystemPromptAction::Prepend(prefix) => {
            let new_content = format!("{}\n\n{}", prefix, current_content);
            req.system = Some(SystemPrompt::String(new_content));
        },
        SystemPromptAction::Append(suffix) => {
            let new_content = format!("{}\n\n{}", current_content, suffix);
            req.system = Some(SystemPrompt::String(new_content));
        },
        SystemPromptAction::MoveToUser(replacement) => {
            // 1. Set system prompt to replacement (or empty)
            if let Some(r) = replacement {
                req.system = Some(SystemPrompt::String(r.clone()));
            } else {
                req.system = None;
            }

            // 2. Prepend old content to first user message
            if let Some(first_msg) = req.messages.iter_mut().find(|m| m.role == "user") {
                match &mut first_msg.content {
                    AnthropicMessageContent::String(s) => {
                        *s = format!("{}\n\n{}", current_content, s);
                    },
                    AnthropicMessageContent::Blocks(blocks) => {
                        // Prepend a text block
                        blocks.insert(0, AnthropicContentBlock::Text { text: current_content.to_string() });
                    }
                }
            } else {
                // No user message found? This is rare for a chat request.
                // Insert a new user message at the beginning?
                req.messages.insert(0, AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::String(current_content.to_string()),
                });
            }
        }
    }
}
