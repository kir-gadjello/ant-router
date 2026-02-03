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

    // 2. Configuration & Args
    let args: Vec<String> = env::args().collect();
    let mut config_path = "./config.yaml".to_string();
    let mut port = 3000;
    let mut host = "0.0.0.0".to_string();

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
                        port = p;
                    }
                    i += 1;
                }
            }
            "--host" => {
                if i + 1 < args.len() {
                    host = args[i + 1].clone();
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
    if let Ok(p) = env::var("PORT") {
        if let Ok(p_parsed) = p.parse() {
            port = p_parsed;
        }
    }
    if let Ok(h) = env::var("HOST") {
        host = h;
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

    // 3. Auth & Upstream
    let base_url = env::var("ANTHROPIC_PROXY_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api".to_string());

    let custom_url_set = env::var("ANTHROPIC_PROXY_BASE_URL").is_ok();

    let api_key = if !custom_url_set {
        Some(
            env::var("OPENROUTER_API_KEY")
                .expect("OPENROUTER_API_KEY required when using default OpenRouter endpoint"),
        )
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

    // 4. Router
    let app = create_router(state);

    // 5. Server
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
