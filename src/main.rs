use anthropic_bridge::{config::Config, create_router, handlers::AppState};
use anyhow::Result;
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Logging
    let debug_mode = env::var("DEBUG").is_ok();
    let log_level = if debug_mode {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    // 2. Configuration Loading
    let args: Vec<String> = env::args().collect();
    let mut config_path = "./config.yaml".to_string();

    // CLI Args parsing
    let mut cli_port = None;
    let mut cli_host = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                if i + 1 < args.len() {
                    config_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse() {
                        cli_port = Some(p);
                    }
                    i += 1;
                }
            }
            "--host" => {
                if i + 1 < args.len() {
                    cli_host = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    if let Ok(p) = env::var("CONFIG_PATH") {
        config_path = p;
    }

    info!("Loading config from {}", config_path);
    let config = Config::load(&config_path).await?;

    // Ensure log directory exists if logging is enabled
    if config.log_enabled {
        let log_path = config.get_log_path();
        if let Some(parent) = log_path.parent() {
            if !parent.exists() {
                info!("Creating log directory: {:?}", parent);
                fs::create_dir_all(parent)?;
            }
        }
    }

    // 3. Resolve Server Settings (Precedence: Env > CLI > Config > Default)
    let mut port = 3000;
    if let Some(p) = config.server.port {
        port = p;
    }
    if let Some(p) = cli_port {
        port = p;
    }
    if let Ok(p) = env::var("PORT") {
        if let Ok(p_parsed) = p.parse() {
            port = p_parsed;
        }
    }

    let mut host = "0.0.0.0".to_string();
    if let Some(h) = &config.server.host {
        host = h.clone();
    }
    if let Some(h) = cli_host {
        host = h;
    }
    if let Ok(h) = env::var("HOST") {
        host = h;
    }

    // 4. Resolve Upstream Settings
    // Base URL
    let mut base_url = "https://openrouter.ai/api".to_string();
    if let Some(url) = &config.upstream.base_url {
        base_url = url.clone();
    }
    if let Ok(url) = env::var("ANTHROPIC_PROXY_BASE_URL") {
        base_url = url;
    }

    // API Key
    // Logic: If custom URL set via Env, maybe skip key?
    // Spec logic was: "If ANTHROPIC_PROXY_BASE_URL is set, no API key required... Otherwise requires OPENROUTER_API_KEY"
    // We should try to respect that but also support the new config.

    let env_url_set = env::var("ANTHROPIC_PROXY_BASE_URL").is_ok();

    let api_key_var_name = config.upstream.api_key_env_var.as_deref().unwrap_or("OPENROUTER_API_KEY");

    let api_key = if !env_url_set {
        // Try to load key from the specified env var
        match env::var(api_key_var_name) {
            Ok(k) => Some(k),
            Err(_) => {
                // If using default OpenRouter, key is required.
                // However, user might be relying on legacy behavior or custom config.
                // Let's log warning and return None, potentially failing upstream if auth is needed.
                // Or panic if strict. Spec said "Otherwise requires OPENROUTER_API_KEY".
                if base_url.contains("openrouter.ai") {
                     panic!("{} required when using default OpenRouter endpoint", api_key_var_name);
                }
                None
            }
        }
    } else {
        None
    };

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(300))
        .build()?;

    let state = Arc::new(AppState {
        config,
        client,
        base_url,
        api_key,
    });

    // 5. Router
    let app = create_router(state);

    // 6. Server
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
