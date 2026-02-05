#[cfg(test)]
mod tests {
    use crate::config::{SystemPromptOp, SystemPromptRule};
    use crate::protocol::{AnthropicMessageRequest, SystemPrompt};
    use crate::middleware::Middleware;
    use crate::middleware::system_prompt::SystemPromptPatcherMiddleware;

    fn make_req(sys: Option<&str>) -> AnthropicMessageRequest {
        AnthropicMessageRequest {
            model: "test".to_string(),
            messages: vec![],
            max_tokens: None,
            metadata: None,
            stop_sequences: None,
            stream: None,
            system: sys.map(|s| SystemPrompt::String(s.to_string())),
            temperature: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            thinking: None,
        }
    }

    #[test]
    fn test_delete_action() {
        let rules = vec![SystemPromptRule {
            name: "delete_test".to_string(),
            r#match: vec!["todelete".to_string()],
            actions: vec![SystemPromptOp::Delete],
        }];

        let middleware = SystemPromptPatcherMiddleware::new(rules);

        let mut req = make_req(Some("This is text todelete."));
        middleware.on_request(&mut req).unwrap();
        assert!(req.system.is_none());

        let mut req_nomatch = make_req(Some("This is keep."));
        middleware.on_request(&mut req_nomatch).unwrap();
        assert!(matches!(req_nomatch.system, Some(SystemPrompt::String(s)) if s == "This is keep."));
    }
}
