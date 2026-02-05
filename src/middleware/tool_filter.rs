use super::Middleware;
use crate::config::{ToolFilterConfig, glob_to_regex};
use crate::protocol::{AnthropicMessageRequest, AnthropicTool};
use anyhow::Result;
use regex::Regex;

pub struct ToolFilterMiddleware {
    config: Option<ToolFilterConfig>,
}

impl ToolFilterMiddleware {
    pub fn new(config: Option<ToolFilterConfig>) -> Self {
        Self { config }
    }
}

impl Middleware for ToolFilterMiddleware {
    fn on_request(&self, req: &mut AnthropicMessageRequest) -> Result<()> {
        if let Some(config) = &self.config {
            if let Some(tools) = &mut req.tools {
                tools.retain(|tool| {
                    let name = match tool {
                        AnthropicTool::Anthropic(t) => &t.name,
                        AnthropicTool::OpenAI(t) => &t.function.name,
                    };

                    // Deny logic
                    if let Some(deny_list) = &config.deny {
                        for pattern in deny_list {
                            // Try raw regex first
                            if let Ok(re) = Regex::new(pattern) {
                                if re.is_match(name) {
                                    return false; // Denied
                                }
                            } else if let Ok(re) = glob_to_regex(pattern) {
                                // Fallback to glob
                                if re.is_match(name) {
                                    return false; // Denied
                                }
                            }
                        }
                    }

                    // Allow logic
                    if let Some(allow_list) = &config.allow {
                        let mut matched = false;
                        for pattern in allow_list {
                            // Try raw regex first
                            if let Ok(re) = Regex::new(pattern) {
                                if re.is_match(name) {
                                    matched = true;
                                    break;
                                }
                            } else if let Ok(re) = glob_to_regex(pattern) {
                                // Fallback to glob
                                if re.is_match(name) {
                                    matched = true;
                                    break;
                                }
                            }
                        }
                        if !matched {
                            return false; // Not allowed
                        }
                    }

                    true // Kept
                });
            }
        }
        Ok(())
    }
}
