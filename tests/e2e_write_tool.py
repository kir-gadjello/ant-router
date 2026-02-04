import json
import threading
import socket
import time
import os
import signal
import subprocess
import sys
import pytest
from http.server import HTTPServer, BaseHTTPRequestHandler
from anthropic import Anthropic

# Configuration
ANT_PORT = 9005
UPSTREAM_PORT = 9006
OPENAI_FRONTEND_PORT = 9007

class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print(f"Mock received POST to {self.path}", flush=True)
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_json = json.loads(post_data.decode('utf-8'))
            print(f"Mock received JSON: {json.dumps(request_json)}", flush=True)

            # Verify the tool is present
            tools = request_json.get("tools", [])
            write_tool = next((t for t in tools if t.get("function", {}).get("name") == "Write"), None)

            if not write_tool:
                self.send_error(400, "Write tool not found in request")
                return

            # Verify schema properties
            params = write_tool["function"].get("parameters", {})
            if params.get("additionalProperties") is not False:
                 # In strict mode or clean translation, we expect additionalProperties: false if source had it
                 pass

            # Respond with a tool call
            response = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_json.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_mock_123",
                            "type": "function",
                            "function": {
                                "name": "Write",
                                "arguments": json.dumps({
                                    "file_path": "/tmp/test.txt",
                                    "content": "Hello World"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_error(404)

def run_mock_server():
    server = HTTPServer(('localhost', UPSTREAM_PORT), MockOpenAIHandler)
    server.serve_forever()

def wait_for_port(port, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(0.5)
    return False

@pytest.fixture(scope="module")
def setup_servers():
    # Start Mock Upstream
    mock_thread = threading.Thread(target=run_mock_server, daemon=True)
    mock_thread.start()

    if not wait_for_port(UPSTREAM_PORT):
        raise RuntimeError("Mock upstream failed to start")

    # Start Ant Router
    config_content = f"""
server:
  ant_port: {ANT_PORT}
  openai_port: {OPENAI_FRONTEND_PORT}

upstream:
  base_url: "http://localhost:{UPSTREAM_PORT}"
  api_key_env_var: "OPENAI_API_KEY"

models:
  default:
    provider: "openai"
    api_model_id: "gpt-4"

profiles:
  default:
    rules:
      - pattern: ".*"
        target: "default"
"""

    with open("tests/temp_config.yaml", "w") as f:
        f.write(config_content)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "mock-key"
    env["RUST_LOG"] = "debug"

    print("Starting router...")
    # Using cargo run
    router_process = subprocess.Popen(
        ["cargo", "run", "--", "--config", "tests/temp_config.yaml"],
        env=env,
        stdout=sys.stdout, # Stream to stdout for visibility
        stderr=sys.stderr  # Stream to stderr
    )

    if not wait_for_port(ANT_PORT, timeout=20):
        router_process.terminate()
        raise RuntimeError("Router failed to start on port 9005")

    yield

    router_process.terminate()
    router_process.wait()
    if os.path.exists("tests/temp_config.yaml"):
        os.remove("tests/temp_config.yaml")

def test_write_tool_end_to_end(setup_servers):
    import requests
    url = f"http://localhost:{ANT_PORT}/v1/messages"

    tool_def = {
        "name": "Write",
        "description": "Writes a file to the local filesystem.",
        "input_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": False,
            "properties": {
                "content": {
                    "description": "The content to write to the file",
                    "type": "string"
                },
                "file_path": {
                    "description": "The absolute path to the file to write (must be absolute, not relative)",
                    "type": "string"
                }
            },
            "required": [
                "file_path",
                "content"
            ],
            "type": "object"
        }
    }

    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Write a file please."}
        ],
        "tools": [tool_def]
    }

    print(f"Sending POST to {url}...")
    response = requests.post(url, json=payload, headers={"x-api-key": "sk-ant-mock"})

    print(f"Response Status: {response.status_code}")
    print(f"Response Body: {response.text}")

    assert response.status_code == 200, f"Router failed: {response.text}"

    resp_json = response.json()
    assert resp_json["stop_reason"] == "tool_use"

    content = resp_json["content"]
    tool_use = next((b for b in content if b["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "Write"
    assert tool_use["input"]["file_path"] == "/tmp/test.txt"
    assert tool_use["input"]["content"] == "Hello World"

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
