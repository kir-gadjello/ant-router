use anthropic_bridge::{config::Config, create_router, create_openai_router, handlers::AppState};
use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

// Embed default config
const DEFAULT_CONFIG_CONTENT: &str = include_str!("../config.default.yaml");

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
    let mut cli_profile = None;
    let mut verbose = false;
    let mut tool_verbose = false;
    let mut record = false;

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
            "--profile" | "-P" => {
                if i + 1 < args.len() {
                    cli_profile = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "-v" | "--verbose" => {
                verbose = true;
            }
            "-tv" => {
                tool_verbose = true;
            }
            "--record" => {
                record = true;
            }
            _ => {}
        }
        i += 1;
    }

    if let Ok(p) = env::var("CONFIG_PATH") {
        config_path = p;
    }

    // Onboarding Logic: Check if config exists, if not and interactive, prompt creation.
    if !Path::new(&config_path).exists() {
        // Check fallback path
        if let Ok(home) = env::var("HOME") {
            let fallback_path = Path::new(&home).join(".ant-router").join("config.yaml");
            if fallback_path.exists() {
                config_path = fallback_path.to_string_lossy().to_string();
            } else if atty::is(atty::Stream::Stdin) {
                // Interactive and no config found anywhere. Prompt user.
                println!("No configuration file found at '{}' or '~/.ant-router/config.yaml'.", config_path);
                print!("Would you like to create a default config at ~/.ant-router/config.yaml? [Y/n] ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim().to_lowercase();

                if input == "y" || input == "yes" || input.is_empty() {
                    let target_dir = Path::new(&home).join(".ant-router");
                    fs::create_dir_all(&target_dir).context("Failed to create config directory")?;
                    let target_path = target_dir.join("config.yaml");
                    fs::write(&target_path, DEFAULT_CONFIG_CONTENT).context("Failed to write default config")?;
                    println!("Created default config at {:?}", target_path);

                    // Switch to using the new config
                    config_path = target_path.to_string_lossy().to_string();
                }
            }
        }
    }

    info!("Loading config from {}", config_path);
    let mut config = Config::load(&config_path).await?;

    // Override profile from CLI or environment (Precedence: CLI > Env > Config)
    if let Some(profile) = cli_profile {
        info!("Overriding profile from CLI: {}", profile);
        config.current_profile = profile;
    } else if let Ok(profile) = env::var("PROFILE") {
        info!("Overriding profile from PROFILE env var: {}", profile);
        config.current_profile = profile;
    }

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
    // Primary/Fallback Port
    let mut primary_port = 3000;
    if let Some(p) = config.server.port {
        primary_port = p;
    }
    if let Some(p) = cli_port {
        primary_port = p;
    }
    if let Ok(p) = env::var("PORT") {
        if let Ok(p_parsed) = p.parse() {
            primary_port = p_parsed;
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

    // Resolve specific ports
    let ant_port = config.server.ant_port.unwrap_or(primary_port);
    let openai_port = config.server.openai_port;

    info!("Resolved Anthropic Port: {}", ant_port);
    if let Some(op) = openai_port {
        info!("Resolved OpenAI Port: {}", op);
    } else {
        info!("OpenAI Frontend Disabled (not configured)");
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
    let env_url_set = env::var("ANTHROPIC_PROXY_BASE_URL").is_ok();
    let api_key_var_name = config.upstream.api_key_env_var.as_deref().unwrap_or("OPENROUTER_API_KEY");

    let api_key = if !env_url_set {
        match env::var(api_key_var_name) {
            Ok(k) => Some(k),
            Err(_) => {
                if base_url.contains("openrouter.ai") {
                     // We don't panic here to allow starting up for help/health check, but warn heavily
                     warn!("{} required when using default OpenRouter endpoint. Requests may fail.", api_key_var_name);
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
        verbose,
        tool_verbose,
        record,
        tools_reported: AtomicBool::new(false),
    });

    // 5. Start Servers
    let mut handles = vec![];

    // Anthropic Server
    let ant_app = create_router(state.clone());
    let ant_addr: SocketAddr = format!("{}:{}", host, ant_port).parse()?;
    info!("Starting Anthropic Proxy on {}", ant_addr);

    // We bind listeners before spawning to fail early if port in use
    let ant_listener = tokio::net::TcpListener::bind(ant_addr).await?;
    handles.push(tokio::spawn(async move {
        if let Err(e) = axum::serve(ant_listener, ant_app).await {
            error!("Anthropic Server Error: {}", e);
        }
    }));

    // OpenAI Server
    if let Some(op) = openai_port {
        let openai_app = create_openai_router(state.clone());
        let openai_addr: SocketAddr = format!("{}:{}", host, op).parse()?;
        info!("Starting OpenAI Proxy on {}", openai_addr);

        let openai_listener = tokio::net::TcpListener::bind(openai_addr).await?;
        handles.push(tokio::spawn(async move {
            if let Err(e) = axum::serve(openai_listener, openai_app).await {
                error!("OpenAI Server Error: {}", e);
            }
        }));
    }

    // Wait for all servers
    if handles.is_empty() {
        warn!("No servers started!");
    } else {
        // We wait for any of them to finish (likely with error) or run forever
        // futures::future::select_all might be better, or just await them all
        for h in handles {
            let _ = h.await;
        }
    }

    Ok(())
}
