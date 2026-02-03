use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

// ==================================================================================
// CONFIG TYPES
// ==================================================================================

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    // Current profile name (legacy/existing routing usage, though Spec doesn't strictly use "profile" for routing,
    // it seems we are keeping it or integrating it. The Spec focuses on UMCP model definitions.
    // We will keep `current_profile` for backward compatibility or routing entry point if needed,
    // but primarily we load the UMCP structure.)
    #[serde(default)]
    pub current_profile: String,

    // Legacy support
    #[serde(default)]
    pub profiles: HashMap<String, Profile>,

    // UMCP Fields
    #[serde(default)]
    pub defaults: Option<Defaults>,
    #[serde(default)]
    pub providers: HashMap<String, Provider>,
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,

    // New Routing Fields (from prompt requirement)
    // These act as specific overrides/pointers for logic handlers
    pub ant_vision_model: Option<String>,
    pub ant_vision_reasoning_model: Option<String>,

    // Logging Configuration
    #[serde(default = "default_log_enabled")]
    pub log_enabled: bool,
    pub log_file: Option<String>,

    // Server Configuration
    #[serde(default)]
    pub server: ServerConfig,

    // Default Upstream Configuration (Global)
    #[serde(default)]
    pub upstream: UpstreamConfig,
}

fn default_log_enabled() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ServerConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct UpstreamConfig {
    pub base_url: Option<String>,
    pub api_key_env_var: Option<String>, // Defines WHICH env var holds the key, defaulting to OPENROUTER_API_KEY
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

// --- UMCP Structs ---

#[derive(Debug, Deserialize, Clone, Default)]
pub struct Defaults {
    #[serde(default)]
    pub context: Option<ContextConstraints>,
    #[serde(default)]
    pub capabilities: Option<Capabilities>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Provider {
    pub base_url: String,
    pub auth_header: Option<String>,
    pub auth_prefix: Option<String>,
    #[serde(default)]
    pub default_headers: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct ModelConfig {
    #[serde(default)]
    pub extends: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub api_model_id: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,

    #[serde(default)]
    pub context: Option<ContextConstraints>,
    #[serde(default)]
    pub capabilities: Option<Capabilities>,

    #[serde(default)]
    pub api_params: Option<ApiParams>,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct ContextConstraints {
    pub window: Option<u32>,
    pub max_output: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct Capabilities {
    #[serde(default)]
    pub vision: Option<bool>,
    #[serde(default)]
    pub tools: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<ReasoningCapability>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(untagged)]
pub enum ReasoningCapability {
    Bool(bool),
    Complex {
        enabled: bool,
        effort_level: Option<String>,
    },
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct ApiParams {
    pub timeout: Option<u32>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub extra_body: HashMap<String, Value>,
}

impl Config {
    pub async fn load(path: &str) -> Result<Self> {
        let mut config_path = PathBuf::from(path);

        // Fallback check: $HOME/.ant-router/config.yaml
        if !config_path.exists() {
            if let Ok(home) = env::var("HOME") {
                let fallback_path = Path::new(&home).join(".ant-router").join("config.yaml");
                if fallback_path.exists() {
                    debug!(
                        "Config not found at {:?}, using fallback at {:?}",
                        config_path, fallback_path
                    );
                    config_path = fallback_path;
                }
            }
        }

        if !config_path.exists() {
            warn!("Config file not found at {:?}, using defaults", config_path);
            return Ok(Config::default());
        }

        let content = tokio::fs::read_to_string(&config_path)
            .await
            .context("Failed to read config file")?;

        let mut config: Config =
            serde_yaml::from_str(&content).context("Failed to parse config file")?;

        // Resolve Model Inheritance (Deep Merge)
        config.resolve_models()?;

        Ok(config)
    }

    pub fn get_log_path(&self) -> PathBuf {
        if let Some(path_str) = &self.log_file {
            PathBuf::from(path_str)
        } else if let Ok(home) = env::var("HOME") {
            Path::new(&home)
                .join(".ant-router")
                .join("logs")
                .join(".log.jsonl")
        } else {
            PathBuf::from(".ant-router/logs/.log.jsonl")
        }
    }

    fn resolve_models(&mut self) -> Result<()> {
        // Simple resolution:
        // We need to resolve models that use `extends`.
        // We can do a topological sort or just iterative resolution since usually chain depth is low.
        // Let's use a safe iterative approach with loop detection or max depth.

        let mut resolved_models = HashMap::new();
        let model_keys: Vec<String> = self.models.keys().cloned().collect();

        for key in model_keys {
            let resolved = self.resolve_single_model(&key, &mut Vec::new())?;
            resolved_models.insert(key, resolved);
        }

        self.models = resolved_models;
        Ok(())
    }

    fn resolve_single_model(&self, key: &str, stack: &mut Vec<String>) -> Result<ModelConfig> {
        if stack.contains(&key.to_string()) {
            return Err(anyhow::anyhow!("Circular dependency detected: {:?}", stack));
        }

        let raw_model = self
            .models
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Model {} not found", key))?;

        if let Some(parent_key) = &raw_model.extends {
            stack.push(key.to_string());
            let parent = self.resolve_single_model(parent_key, stack)?;
            stack.pop();

            // Merge parent + child (child overrides)
            Ok(merge_models(parent, raw_model.clone()))
        } else {
            Ok(raw_model.clone())
        }
    }

    // New resolution method compatible with UMCP but also supporting legacy profiles if needed
    pub fn resolve_model_alias(&self, alias: &str) -> Option<&ModelConfig> {
        // 1. Direct key match
        if let Some(m) = self.models.get(alias) {
            return Some(m);
        }
        // 2. Alias search
        self.models
            .values()
            .find(|&m| m.aliases.contains(&alias.to_string()))
            .map(|v| v as _)
    }

    // Helper to get wire model ID from a resolved config
    pub fn get_wire_model_id(&self, model_conf: &ModelConfig) -> String {
        model_conf
            .api_model_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string())
    }

    // Restore legacy resolve_model for backward compatibility
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

// Merging Logic
fn merge_models(parent: ModelConfig, child: ModelConfig) -> ModelConfig {
    ModelConfig {
        extends: child.extends, // Keep child's extends pointer (though resolved)
        provider: child.provider.or(parent.provider),
        api_model_id: child.api_model_id.or(parent.api_model_id),
        aliases: [parent.aliases, child.aliases].concat(), // Union/Concat aliases

        context: merge_context(parent.context, child.context),
        capabilities: merge_capabilities(parent.capabilities, child.capabilities),
        api_params: merge_api_params(parent.api_params, child.api_params),
    }
}

fn merge_context(
    parent: Option<ContextConstraints>,
    child: Option<ContextConstraints>,
) -> Option<ContextConstraints> {
    match (parent, child) {
        (None, None) => None,
        (Some(p), None) => Some(p),
        (None, Some(c)) => Some(c),
        (Some(p), Some(c)) => Some(ContextConstraints {
            window: c.window.or(p.window),
            max_output: c.max_output.or(p.max_output),
        }),
    }
}

fn merge_capabilities(
    parent: Option<Capabilities>,
    child: Option<Capabilities>,
) -> Option<Capabilities> {
    match (parent, child) {
        (None, None) => None,
        (Some(p), None) => Some(p),
        (None, Some(c)) => Some(c),
        (Some(p), Some(c)) => Some(Capabilities {
            vision: c.vision.or(p.vision),
            tools: c.tools.or(p.tools),
            reasoning: c.reasoning.or(p.reasoning), // Simple overwrite for reasoning for now
        }),
    }
}

fn merge_api_params(parent: Option<ApiParams>, child: Option<ApiParams>) -> Option<ApiParams> {
    match (parent, child) {
        (None, None) => None,
        (Some(p), None) => Some(p),
        (None, Some(c)) => Some(c),
        (Some(p), Some(c)) => {
            let mut headers = p.headers.clone();
            headers.extend(c.headers);

            // Deep merge extra_body
            let extra_body = deep_merge_json(p.extra_body, c.extra_body);

            Some(ApiParams {
                timeout: c.timeout.or(p.timeout),
                headers,
                extra_body,
            })
        }
    }
}

fn deep_merge_json(
    base: HashMap<String, Value>,
    override_map: HashMap<String, Value>,
) -> HashMap<String, Value> {
    let mut result = base;
    for (k, v) in override_map {
        match (result.get_mut(&k), v) {
            (Some(Value::Object(base_obj)), Value::Object(override_obj)) => {
                // If both are objects, merge recursively
                let mut base_map = HashMap::new();
                for (bk, bv) in base_obj.iter() {
                    base_map.insert(bk.clone(), bv.clone());
                }
                let mut override_map_inner = HashMap::new();
                for (ok, ov) in override_obj {
                    override_map_inner.insert(ok, ov);
                }

                let merged = deep_merge_json(base_map, override_map_inner);
                *base_obj = serde_json::to_value(merged)
                    .unwrap()
                    .as_object()
                    .unwrap()
                    .clone();
            }
            (Some(existing), new_val) => {
                *existing = new_val;
            }
            (None, new_val) => {
                result.insert(k, new_val);
            }
        }
    }
    result
}

// Legacy glob regex helper (kept for profile support)
pub fn glob_to_regex(pattern: &str) -> Result<Regex, regex::Error> {
    let escaped = regex::escape(pattern);
    let regex_pattern = escaped.replace("\\*", ".*");
    let final_pattern = format!("(?i)^{}$", regex_pattern);
    Regex::new(&final_pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_merge_inheritance() {
        let parent = ModelConfig {
            api_model_id: Some("base".to_string()),
            api_params: Some(ApiParams {
                timeout: Some(10),
                extra_body: HashMap::from([
                    (
                        "stream_options".to_string(),
                        serde_json::json!({"include_usage": true}),
                    ),
                    ("nested".to_string(), serde_json::json!({"a": 1})),
                ]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let child = ModelConfig {
            extends: Some("base".to_string()),
            api_model_id: Some("child".to_string()),
            api_params: Some(ApiParams {
                extra_body: HashMap::from([("nested".to_string(), serde_json::json!({"b": 2}))]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let merged = merge_models(parent, child);

        assert_eq!(merged.api_model_id, Some("child".to_string()));
        assert_eq!(merged.api_params.as_ref().unwrap().timeout, Some(10));

        let body = &merged.api_params.unwrap().extra_body;
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["nested"]["a"], 1);
        assert_eq!(body["nested"]["b"], 2);
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
            ..Default::default() // Important: fill other fields with default
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

// Add Default impl for Config to support tests
impl Default for Config {
    fn default() -> Self {
        Self {
            current_profile: "default".to_string(),
            profiles: HashMap::new(),
            defaults: None,
            providers: HashMap::new(),
            models: HashMap::new(),
            ant_vision_model: None,
            ant_vision_reasoning_model: None,
            log_enabled: true,
            log_file: None,
            server: ServerConfig::default(),
            upstream: UpstreamConfig::default(),
        }
    }
}
