use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use tracing::{debug, warn};

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub current_profile: String,
    pub profiles: HashMap<String, Profile>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Profile {
    pub rules: Vec<Rule>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Rule {
    pub pattern: String,
    pub target: String,
    pub reasoning_target: Option<String>,
}

impl Config {
    pub async fn load(path: &str) -> Result<Self> {
        let config_path = Path::new(path);

        if !config_path.exists() {
            warn!("Config file not found at {:?}, using defaults", config_path);
            // Return a default config or fail? Spec says:
            // "If missing: log warning, continue with env vars only"
            // But Config struct requires current_profile.
            // I'll create a dummy config that won't match anything, so fallback to env vars works.
            return Ok(Config {
                current_profile: "default".to_string(),
                profiles: HashMap::new(),
            });
        }

        let content = tokio::fs::read_to_string(config_path)
            .await
            .context("Failed to read config file")?;

        let config: Config =
            serde_yaml::from_str(&content).context("Failed to parse config file")?;

        Ok(config)
    }

    pub fn resolve_model(&self, input_model: &str, thinking: bool) -> String {
        // 1. Try to match in current profile
        if let Some(profile) = self.profiles.get(&self.current_profile) {
            for rule in &profile.rules {
                if let Ok(regex) = glob_to_regex(&rule.pattern) {
                    if regex.is_match(input_model) {
                        if thinking {
                            if let Some(ref target) = rule.reasoning_target {
                                debug!(
                                    "Matched rule for model '{}', using reasoning target '{}'",
                                    input_model, target
                                );
                                return target.clone();
                            }
                        }
                        debug!(
                            "Matched rule for model '{}', using target '{}'",
                            input_model, rule.target
                        );
                        return rule.target.clone();
                    }
                } else {
                    warn!("Invalid regex pattern from glob: {}", rule.pattern);
                }
            }
        }

        // 2. Env vars
        if thinking {
            if let Ok(model) = env::var("REASONING_MODEL") {
                debug!("Using REASONING_MODEL env var: {}", model);
                return model;
            }
        }

        if let Ok(model) = env::var("COMPLETION_MODEL") {
            debug!("Using COMPLETION_MODEL env var: {}", model);
            return model;
        }

        // 3. Hardcoded default
        let default_model = "google/gemini-2.0-pro-exp-02-05:free";
        debug!("Using default model: {}", default_model);
        default_model.to_string()
    }
}

fn glob_to_regex(pattern: &str) -> Result<Regex, regex::Error> {
    // Escape regex special chars: .+?^${}()|[]\
    let escaped = regex::escape(pattern);
    // Replace escaped * (\*) with .*
    // Note: regex::escape escapes * as \*
    let regex_pattern = escaped.replace("\\*", ".*");
    // Wrap with ^...$ and make case insensitive
    let final_pattern = format!("(?i)^{}$", regex_pattern);

    Regex::new(&final_pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_to_regex() {
        let regex = glob_to_regex("claude-3-5-*").unwrap();
        assert!(regex.is_match("claude-3-5-sonnet"));
        assert!(regex.is_match("Claude-3-5-Sonnet")); // Case insensitive
        assert!(!regex.is_match("claude-3-opus"));

        let regex_all = glob_to_regex("*").unwrap();
        assert!(regex_all.is_match("anything"));
    }

    #[test]
    fn test_resolve_model() {
        let config = Config {
            current_profile: "test".to_string(),
            profiles: HashMap::from([(
                "test".to_string(),
                Profile {
                    rules: vec![
                        Rule {
                            pattern: "claude*".to_string(),
                            target: "openai/gpt-4o".to_string(),
                            reasoning_target: Some("openai/o1".to_string()),
                        },
                        Rule {
                            pattern: "*".to_string(),
                            target: "catch-all".to_string(),
                            reasoning_target: None,
                        },
                    ],
                },
            )]),
        };

        // Match claude, no thinking
        assert_eq!(
            config.resolve_model("claude-3-opus", false),
            "openai/gpt-4o"
        );
        // Match claude, thinking
        assert_eq!(config.resolve_model("claude-3-opus", true), "openai/o1");

        // Match catch-all
        assert_eq!(config.resolve_model("llama-3", false), "catch-all");
    }
}
