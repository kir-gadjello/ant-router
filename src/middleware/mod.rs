use crate::protocol::{AnthropicMessageRequest, AnthropicMessageResponse, AnthropicStreamEvent};
use anyhow::Result;
use futures::Stream;
use std::pin::Pin;

pub mod system_prompt;
pub mod tool_enforcer;

pub type StreamBox = Pin<Box<dyn Stream<Item = Result<AnthropicStreamEvent, anyhow::Error>> + Send>>;

pub trait Middleware: Send + Sync {
    /// Transform the request before it is sent to the upstream (and before conversion to OpenAI format)
    fn on_request(&self, _req: &mut AnthropicMessageRequest) -> Result<()> {
        Ok(())
    }

    /// Transform the response after it is received from upstream (and after conversion from OpenAI format)
    /// This is called for non-streaming responses.
    fn on_response(&self, _resp: &mut AnthropicMessageResponse) -> Result<()> {
        Ok(())
    }

    /// Transform the response stream.
    /// The default implementation returns the stream as-is.
    fn transform_stream(&self, stream: StreamBox) -> StreamBox {
        stream
    }
}
