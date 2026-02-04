use crate::protocol::{AnthropicMessageRequest, OpenAIChatCompletionRequest};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use lazy_static::lazy_static;

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

#[derive(Serialize)]
#[serde(tag = "event", content = "data")]
pub enum TraceEvent {
    FrontendRequest {
        id: String,
        timestamp: DateTime<Utc>,
        payload: AnthropicMessageRequest,
    },
    UpstreamRequest {
        id: String,
        timestamp: DateTime<Utc>,
        payload: OpenAIChatCompletionRequest,
    },
    UpstreamResponse {
        id: String,
        timestamp: DateTime<Utc>,
        payload: Value, // OpenAIChatCompletionResponse or Chunk
    },
    FrontendResponse {
        id: String,
        timestamp: DateTime<Utc>,
        payload: Value, // AnthropicMessageResponse or StreamEvent
    }
}

lazy_static! {
    static ref TRACE_FILE: Mutex<Option<PathBuf>> = Mutex::new(None);
}

pub fn set_trace_file(path: PathBuf) {
    if let Ok(mut guard) = TRACE_FILE.lock() {
        *guard = Some(path);
    }
}

pub fn log_trace(event: TraceEvent) {
    let path = {
        let guard = TRACE_FILE.lock().unwrap();
        guard.clone()
    };

    if let Some(p) = path {
        if let Ok(line) = serde_json::to_string(&event) {
            // Best effort write
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&p) {
                let _ = writeln!(file, "{}", line);
            }
        }
    }
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
