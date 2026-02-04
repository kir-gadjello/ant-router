use super::Middleware;
use crate::protocol::{AnthropicMessageRequest, SystemBlock, SystemPrompt};
use anyhow::Result;
use std::collections::HashMap;

pub struct SystemPromptPatcherMiddleware {
    pub prepend: Option<String>,
    pub append: Option<String>,
}

impl SystemPromptPatcherMiddleware {
    pub fn new(prepend: Option<String>, append: Option<String>) -> Self {
        Self { prepend, append }
    }
}

impl Middleware for SystemPromptPatcherMiddleware {
    fn on_request(&self, req: &mut AnthropicMessageRequest) -> Result<()> {
        if self.prepend.is_none() && self.append.is_none() {
            return Ok(());
        }

        match &mut req.system {
            Some(SystemPrompt::String(s)) => {
                if let Some(pre) = &self.prepend {
                    *s = format!("{}\n\n{}", pre, s);
                }
                if let Some(post) = &self.append {
                    *s = format!("{}\n\n{}", s, post);
                }
            }
            Some(SystemPrompt::Array(blocks)) => {
                // If it's an array, we prepend/append text blocks
                if let Some(pre) = &self.prepend {
                    blocks.insert(0, SystemBlock {
                        r#type: "text".to_string(),
                        text: pre.clone(),
                        other: HashMap::new(),
                    });
                }
                if let Some(post) = &self.append {
                    blocks.push(SystemBlock {
                        r#type: "text".to_string(),
                        text: post.clone(),
                        other: HashMap::new(),
                    });
                }
            }
            None => {
                let mut text = String::new();
                if let Some(pre) = &self.prepend {
                    text.push_str(pre);
                }
                if let Some(post) = &self.append {
                    if !text.is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(post);
                }
                if !text.is_empty() {
                    req.system = Some(SystemPrompt::String(text));
                }
            }
        }

        Ok(())
    }
}
