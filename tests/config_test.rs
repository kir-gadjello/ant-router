
#[tokio::test]
async fn test_default_config_parsing() {
    use anthropic_bridge::config::Config;

    // Test parsing the generated config.default.yaml
    let content = std::fs::read_to_string("config.default.yaml").unwrap();
    let config: Config = serde_yaml::from_str(&content).unwrap();

    assert_eq!(config.server.host, Some("127.0.0.1".to_string()));
    assert_eq!(config.server.port, Some(3000));
    assert_eq!(config.upstream.base_url, Some("https://openrouter.ai/api".to_string()));
    assert!(config.log_enabled);

    // Check legacy profile
    assert!(config.profiles.contains_key("stepfun"));

    // Check UMCP model
    assert!(config.models.contains_key("glm-4d6-v"));
}
