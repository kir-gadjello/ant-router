use crate::protocol::AnthropicMessageRequest;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Serialize)]
struct LogEntry {
    _metadata: LogMetadata,
    #[serde(flatten)]
    request: AnthropicMessageRequest,
}

#[derive(Serialize)]
struct LogMetadata {
    timestamp: DateTime<Utc>,
    version: String,
}

#[derive(Serialize)]
struct InteractionEntry {
    timestamp: DateTime<Utc>,
    request: AnthropicMessageRequest,
    response: Value,
}

pub fn log_request(req: &AnthropicMessageRequest, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).context("Failed to create log directory")?;
        }
    }

    let entry = LogEntry {
        _metadata: LogMetadata {
            timestamp: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        request: req.clone(),
    };

    let line = serde_json::to_string(&entry).context("Failed to serialize log entry")?;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .context("Failed to open log file")?;

    writeln!(file, "{}", line).context("Failed to write log entry")?;

    Ok(())
}

pub fn record_interaction(req: &AnthropicMessageRequest, resp: &Value) -> Result<()> {
    let entry = InteractionEntry {
        timestamp: Utc::now(),
        request: req.clone(),
        response: resp.clone(),
    };

    let line = serde_json::to_string(&entry).context("Failed to serialize interaction entry")?;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("recorded_logs.jsonl")
        .context("Failed to open recorded_logs.jsonl")?;

    writeln!(file, "{}", line).context("Failed to write interaction entry")?;

    Ok(())
}
