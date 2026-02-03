pub mod config;
pub mod handlers;
pub mod protocol;
pub mod transformer;

use axum::{
    routing::{get, post},
    Router,
};
use handlers::{handle_messages, health_check, AppState};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/messages", post(handle_messages))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
