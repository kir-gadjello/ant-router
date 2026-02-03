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
    #[serde(default)]
    pub current_profile: String,

    // Core feature: Profiles map to Models
    #[serde(default)]
    pub profiles: HashMap<String, Profile>,

    #[serde(default)]
    pub defaults: Option<Defaults>,
    #[serde(default)]
    pub providers: HashMap<String, Provider>,
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,

    // Config flags
    #[serde(default = "default_log_enabled")]
    pub log_enabled: bool,
    pub log_file: Option<String>,
    #[serde(default)]
    pub no_ant: bool,

    #[serde(default)]
    pub server: ServerConfig,
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
    pub api_key_env_var: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Profile {
    pub rules: Vec<Rule>,
    // Profile-specific vision routing
    pub ant_vision_model: Option<String>,
    pub ant_vision_reasoning_model: Option<String>,
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
        // Validate that all profile targets map to existing models
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
            if let Some(vision_model) = &profile.ant_vision_model {
                if !self.models.contains_key(vision_model) {
                    return Err(anyhow::anyhow!(
                        "Profile '{}' ant_vision_model '{}' unknown",
                        profile_name,
                        vision_model
                    ));
                }
            }
            if let Some(vision_reasoning) = &profile.ant_vision_reasoning_model {
                if !self.models.contains_key(vision_reasoning) {
                    return Err(anyhow::anyhow!(
                        "Profile '{}' ant_vision_reasoning_model '{}' unknown",
                        profile_name,
                        vision_reasoning
                    ));
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

// Merging Logic
fn merge_models(parent: ModelConfig, child: ModelConfig) -> ModelConfig {
    ModelConfig {
        extends: child.extends,
        provider: child.provider.or(parent.provider),
        api_model_id: child.api_model_id.or(parent.api_model_id),
        aliases: [parent.aliases, child.aliases].concat(),
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
            reasoning: c.reasoning.or(p.reasoning),
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
        }
    }
}
