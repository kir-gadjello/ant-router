use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize, Deserializer};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use tracing::warn;

// ==================================================================================
// CONFIG TYPES
// ==================================================================================

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_profile")]
    pub current_profile: String,

    #[serde(default)]
    pub profiles: HashMap<String, Profile>,

    #[serde(default)]
    pub defaults: Option<Defaults>,
    #[serde(default)]
    pub providers: HashMap<String, Provider>,
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,

    #[serde(default = "default_log_enabled")]
    pub log_enabled: bool,
    pub log_file: Option<String>,
    #[serde(default)]
    pub no_ant: bool,

    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub upstream: UpstreamConfig,

    #[serde(default)]
    pub tool_filters: Option<ToolFilterConfig>,
    #[serde(default)]
    pub system_prompts: Vec<SystemPromptRule>,
    #[serde(default = "default_true")]
    pub enable_exit_tool: bool,
    #[serde(default)]
    pub debug_tools: bool,
    #[serde(default)]
    pub trace_file: Option<String>,
}

fn default_log_enabled() -> bool {
    true
}

fn default_true() -> bool {
    true
}

fn default_profile() -> String {
    "default".to_string()
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ServerConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub ant_port: Option<u16>,
    pub openai_port: Option<u16>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct UpstreamConfig {
    pub base_url: Option<String>,
    pub api_key_env_var: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Profile {
    pub rules: Vec<Rule>,
    #[serde(default)]
    pub tool_filters: Option<ToolFilterConfig>,
    #[serde(default)]
    pub system_prompts: Option<Vec<SystemPromptRule>>,
    #[serde(default)]
    pub preprocess: Option<PreprocessConfig>,
    #[serde(default)]
    pub enable_exit_tool: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Rule {
    pub pattern: String,
    #[serde(default)]
    pub match_features: Vec<String>,
    pub target: String,
    pub reasoning_target: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ToolFilterConfig {
    pub allow: Option<Vec<String>>,
    pub deny: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SystemPromptRule {
    pub name: String,
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    pub r#match: Vec<String>,
    pub actions: Vec<SystemPromptOp>,
}

fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<String>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::String(s) => Ok(vec![s]),
        StringOrVec::Vec(v) => Ok(v),
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum SystemPromptOp {
    #[serde(rename = "replace")]
    Replace {
        pattern: String,
        with: String,
    },
    #[serde(rename = "prepend")]
    Prepend {
        value: String,
    },
    #[serde(rename = "append")]
    Append {
        value: String,
    },
    #[serde(rename = "move_to_user")]
    MoveToUser {
        forced_system_prompt: Option<String>,
        prefix: Option<String>,
        suffix: Option<String>,
    },
    #[serde(rename = "delete")]
    Delete,
}

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
    pub min_reasoning: Option<ReasoningConfig>,
    
    #[serde(default)]
    pub force_reasoning: Option<ReasoningConfig>,

    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub override_max_tokens: Option<Value>,

    #[serde(default)]
    pub r#override: Option<HashMap<String, Value>>,

    #[serde(default)]
    pub preprocess: Option<PreprocessConfig>,

    #[serde(default)]
    pub api_params: Option<ApiParams>,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct PreprocessConfig {
    #[serde(default)]
    pub merge_system_messages: Option<bool>,
    #[serde(default)]
    pub sanitize_tool_history: Option<bool>,
    #[serde(default)]
    pub max_output_tokens: Option<Value>,
    #[serde(default)]
    pub max_output_cap: Option<u32>,
    #[serde(default)]
    pub disable_parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub strict_tools: Option<bool>,
    #[serde(default)]
    pub clean_web_search: Option<bool>,
    #[serde(default)]
    pub json_repair: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(untagged)]
pub enum ReasoningConfig {
    Bool(bool),
    Level(String), 
    Budget(u32),   
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
    #[serde(default)]
    pub max_context_tokens: Option<Value>,
    #[serde(default)]
    pub max_output_tokens: Option<Value>,
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
    pub retry: Option<RetryConfig>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub backoff_ms: u64,
}

impl Config {
    pub async fn load(path: &str) -> Result<Self> {
        let mut config_path = PathBuf::from(path);

        if !config_path.exists() {
            if let Ok(home) = env::var("HOME") {
                let fallback_path = Path::new(&home).join(".ant-router").join("config.yaml");
                if fallback_path.exists() {
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

        config.resolve_models()?;
        config.validate()?;

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
            Ok(merge_models(parent, raw_model.clone()))
        } else {
            Ok(raw_model.clone())
        }
    }

    fn validate(&self) -> Result<()> {
        for (profile_name, profile) in &self.profiles {
            for rule in &profile.rules {
                if !self.models.contains_key(&rule.target) {
                    return Err(anyhow::anyhow!(
                        "Profile '{}' rule pattern '{}' targets unknown model '{}'",
                        profile_name,
                        rule.pattern,
                        rule.target
                    ));
                }
                if let Some(reasoning_target) = &rule.reasoning_target {
                    if !self.models.contains_key(reasoning_target) {
                        return Err(anyhow::anyhow!(
                            "Profile '{}' rule pattern '{}' reasoning target '{}' unknown",
                            profile_name,
                            rule.pattern,
                            reasoning_target
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_wire_model_id(&self, model_conf: &ModelConfig) -> String {
        model_conf
            .api_model_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string())
    }
}

fn merge_models(parent: ModelConfig, child: ModelConfig) -> ModelConfig {
    ModelConfig {
        extends: child.extends,
        provider: child.provider.or(parent.provider),
        api_model_id: child.api_model_id.or(parent.api_model_id),
        aliases: [parent.aliases, child.aliases].concat(),
        context: merge_context(parent.context, child.context),
        capabilities: merge_capabilities(parent.capabilities, child.capabilities),
        min_reasoning: child.min_reasoning.or(parent.min_reasoning),
        force_reasoning: child.force_reasoning.or(parent.force_reasoning),
        max_tokens: child.max_tokens.or(parent.max_tokens),
        override_max_tokens: child.override_max_tokens.or(parent.override_max_tokens),
        r#override: merge_overrides(parent.r#override, child.r#override),
        preprocess: merge_preprocess(parent.preprocess, child.preprocess),
        api_params: merge_api_params(parent.api_params, child.api_params),
    }
}

fn merge_overrides(
    parent: Option<HashMap<String, Value>>,
    child: Option<HashMap<String, Value>>,
) -> Option<HashMap<String, Value>> {
    match (parent, child) {
        (None, None) => None,
        (Some(p), None) => Some(p),
        (None, Some(c)) => Some(c),
        (Some(p), Some(c)) => Some(deep_merge_json(p, c)),
    }
}

pub fn merge_preprocess(
    parent: Option<PreprocessConfig>,
    child: Option<PreprocessConfig>,
) -> Option<PreprocessConfig> {
    match (parent, child) {
        (None, None) => None,
        (Some(p), None) => Some(p),
        (None, Some(c)) => Some(c),
        (Some(p), Some(c)) => Some(PreprocessConfig {
            merge_system_messages: c.merge_system_messages.or(p.merge_system_messages),
            sanitize_tool_history: c.sanitize_tool_history.or(p.sanitize_tool_history),
            max_output_tokens: c.max_output_tokens.or(p.max_output_tokens),
            max_output_cap: c.max_output_cap.or(p.max_output_cap),
            disable_parallel_tool_calls: c.disable_parallel_tool_calls.or(p.disable_parallel_tool_calls),
            strict_tools: c.strict_tools.or(p.strict_tools),
            clean_web_search: c.clean_web_search.or(p.clean_web_search),
            json_repair: c.json_repair.or(p.json_repair),
        }),
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
            reasoning: c.reasoning.or(p.reasoning),
            max_context_tokens: c.max_context_tokens.or(p.max_context_tokens),
            max_output_tokens: c.max_output_tokens.or(p.max_output_tokens),
        }),
    }
}

pub fn parse_token_count(v: &Value) -> Option<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|u| u as u32),
        Value::String(s) => {
            let s = s.trim().to_lowercase();
            if let Some(stripped) = s.strip_suffix('k') {
                stripped.parse::<f64>().ok().map(|n| (n * 1000.0) as u32)
            } else {
                s.parse::<u32>().ok()
            }
        },
        _ => None
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
            let extra_body = deep_merge_json(p.extra_body, c.extra_body);
            Some(ApiParams {
                timeout: c.timeout.or(p.timeout),
                retry: c.retry.or(p.retry.clone()),
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

pub fn glob_to_regex(pattern: &str) -> Result<Regex, regex::Error> {
    if pattern == ".*" {
        return Regex::new("(?i)^.*$");
    }
    let escaped = regex::escape(pattern);
    let regex_pattern = escaped.replace("\\*", ".*");
    let final_pattern = format!("(?i)^{}$", regex_pattern);
    Regex::new(&final_pattern)
}

impl Default for Config {
    fn default() -> Self {
        Self {
            current_profile: "default".to_string(),
            profiles: HashMap::new(),
            defaults: None,
            providers: HashMap::new(),
            models: HashMap::new(),
            log_enabled: true,
            log_file: None,
            no_ant: false,
            server: ServerConfig::default(),
            upstream: UpstreamConfig::default(),
            tool_filters: None,
            system_prompts: Vec::new(),
            enable_exit_tool: true,
            debug_tools: false,
            trace_file: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_token_count() {
        assert_eq!(parse_token_count(&json!(1000)), Some(1000));
        assert_eq!(parse_token_count(&json!("2000")), Some(2000));
        assert_eq!(parse_token_count(&json!("1k")), Some(1000));
        assert_eq!(parse_token_count(&json!("1.5k")), Some(1500));
        assert_eq!(parse_token_count(&json!("128K")), Some(128000));
        assert_eq!(parse_token_count(&json!("invalid")), None);
    }
}