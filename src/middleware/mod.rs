pub mod system_prompt;
pub mod tool_enforcer;
pub mod tool_filter;

use crate::protocol::{AnthropicMessageRequest, AnthropicMessageResponse, AnthropicStreamEvent};
use anyhow::Result;
use futures::Stream;
use std::pin::Pin;

pub type StreamBox = Pin<Box<dyn Stream<Item = Result<AnthropicStreamEvent, anyhow::Error>> + Send>>;

pub trait Middleware: Send + Sync {
    fn on_request(&self, _req: &mut AnthropicMessageRequest) -> Result<()> {
        Ok(())
    }

    fn on_response(&self, _resp: &mut AnthropicMessageResponse) -> Result<()> {
        Ok(())
    }

    fn transform_stream(&self, stream: StreamBox) -> StreamBox {
        stream
    }
}
