import os
import sys
import time
import subprocess
import threading
import json
import pytest
from http.server import HTTPServer, BaseHTTPRequestHandler
from anthropic import Anthropic

# Mock OpenAI Server
class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data.decode('utf-8'))
            stream = request.get('stream', False)

            # Helper to find last user message
            last_user_content = ""
            for m in reversed(request['messages']):
                if m['role'] == 'user':
                    if isinstance(m['content'], str):
                        last_user_content = m['content']
                    elif isinstance(m['content'], list):
                        # Simple extraction
                        for b in m['content']:
                             if b.get('type') == 'text':
                                 last_user_content += b.get('text', '')
                    break

            # Check if this is the first request (has tools, no tool output)
            is_tool_response = False
            for msg in request['messages']:
                if msg['role'] == 'tool':
                    is_tool_response = True
                    break

            if stream:
                self.send_response(200)
                self.send_header('Content-type', 'text/event-stream')
                self.end_headers()

                if "Exit now" in last_user_content:
                    # Stream ExitTool call
                    # 1. Tool Call Start
                    chunk1 = {
                        "id": "chatcmpl-exit-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_exit_1", "type": "function", "function": {"name": "ExitTool", "arguments": ""}}]}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk1)}\n\n".encode('utf-8'))

                    # 2. Tool Call Args (split to test streaming)
                    chunk2 = {
                        "id": "chatcmpl-exit-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"response\": \"I "}}]}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk2)}\n\n".encode('utf-8'))

                    chunk3 = {
                        "id": "chatcmpl-exit-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "am exiting.\"}"}}]}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk3)}\n\n".encode('utf-8'))

                    # 3. Finish
                    chunk4 = {
                        "id": "chatcmpl-exit-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk4)}\n\n".encode('utf-8'))
                    self.wfile.write(b"data: [DONE]\n\n")

                else:
                    # Normal response stream
                    chunk = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))

                    chunk_finish = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_finish)}\n\n".encode('utf-8'))
                    self.wfile.write(b"data: [DONE]\n\n")

            else:
                # NON-STREAMING RESPONSE
                response = {}

                if "Exit now" in last_user_content:
                    # Return ExitTool call
                    response = {
                        "id": "chatcmpl-exit",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call_exit_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ExitTool",
                                        "arguments": "{\"response\": \"I am exiting.\"}"
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
                    }
                elif not is_tool_response:
                    # 1. First request: Return tool call
                    response = {
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call_weather_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "{\"location\": \"San Francisco\"}"
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
                    }
                else:
                    # 2. Second request: Return final answer
                    tool_msg = next(m for m in request['messages'] if m['role'] == 'tool')

                    if tool_msg['tool_call_id'] != "call_weather_123":
                        print(f"ERROR: Expected tool_call_id 'call_weather_123', got {tool_msg['tool_call_id']}")

                    response = {
                        "id": "chatcmpl-124",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request['model'],
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The weather is 15 degrees.",
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
                    }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_mock_server(port):
    server = HTTPServer(('localhost', port), MockOpenAIHandler)
    server.serve_forever()

@pytest.fixture(scope="module")
def setup_servers():
    # Start Mock Server
    mock_port = 4001
    mock_thread = threading.Thread(target=run_mock_server, args=(mock_port,))
    mock_thread.daemon = True
    mock_thread.start()

    # Start App Server (Cargo run)
    app_port = 3001
    # We use a custom config or env vars
    env = os.environ.copy()
    env['PORT'] = str(app_port)
    env['ANTHROPIC_PROXY_BASE_URL'] = f"http://localhost:{mock_port}"
    env['OPENROUTER_API_KEY'] = "dummy"

    # Create test config
    config_content = """
current_profile: default
profiles:
  default:
    rules:
      - pattern: ".*"
        target: "mock-model"

models:
  "mock-model":
    provider: openai
    api_model_id: "mock-model"
"""
    config_path = "test_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    # Ensure cargo is built
    print("Building project...")
    subprocess.run(["cargo", "build"], check=True)

    print(f"Starting server on port {app_port}...")
    # Use files for logs
    stdout_file = open("server_stdout.log", "w")
    stderr_file = open("server_stderr.log", "w")

    proc = subprocess.Popen(
        ["cargo", "run", "--", "--port", str(app_port), "--config", config_path],
        env=env,
        stdout=stdout_file,
        stderr=stderr_file
    )

    # Wait for server to be ready
    time.sleep(10)

    yield app_port

    # Debug: Check connectivity
    print("Checking /health...")
    subprocess.run(["curl", "-v", f"http://localhost:{app_port}/health"])

    print("Checking /v1/messages POST with JSON...")
    subprocess.run(["curl", "-v", "-X", "POST", f"http://localhost:{app_port}/v1/messages",
                    "-H", "Content-Type: application/json",
                    "-d", '{"model": "test", "messages": [{"role": "user", "content": "hi"}]}'])

    print("Terminating server...")
    proc.terminate()
    proc.wait()
    stdout_file.close()
    stderr_file.close()

    print("--- Server STDOUT ---")
    with open("server_stdout.log", "r") as f:
        print(f.read())
    print("--- Server STDERR ---")
    with open("server_stderr.log", "r") as f:
        print(f.read())

    # Cleanup config
    if os.path.exists(config_path):
        os.remove(config_path)

def test_tool_calling_e2e(setup_servers):
    app_port = setup_servers
    client = Anthropic(
        base_url=f"http://localhost:{app_port}", # Trying without /v1
        api_key="dummy"
    )

    # 1. Send request with tools
    messages = [{"role": "user", "content": "What is the weather in SF?"}]
    tools = [{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }]

    print("Sending first request...")
    resp = client.messages.create(
        model="claude-3-opus-20240229", # Logical ID
        max_tokens=1024,
        messages=messages,
        tools=tools
    )

    # Verify we got a tool use request
    assert resp.stop_reason == "tool_use"
    assert len(resp.content) > 0
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    assert tool_use.name == "get_weather"
    assert tool_use.input == {"location": "San Francisco"}

    print("Got tool use:", tool_use)

    # 2. Send tool result
    messages.append({"role": "assistant", "content": resp.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": "15 degrees"
        }]
    })

    print("Sending second request...")
    resp2 = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=messages,
        tools=tools
    )

    print("Got final response:", resp2.content[0].text)
    assert resp2.content[0].text == "The weather is 15 degrees."

def test_exit_tool_e2e(setup_servers):
    app_port = setup_servers
    client = Anthropic(
        base_url=f"http://localhost:{app_port}", # Trying without /v1
        api_key="dummy"
    )

    # Trigger ExitTool
    messages = [{"role": "user", "content": "Exit now"}]
    tools = [{
        "name": "dummy",
        "description": "dummy",
        "input_schema": {"type": "object", "properties": {}}
    }]

    print("Sending Exit request...")
    resp = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=messages,
        tools=tools
    )

    print("Got response content:", resp.content)
    # Middleware should have transformed ExitTool call into text
    assert resp.stop_reason == "end_turn"
    assert len(resp.content) == 1
    assert resp.content[0].type == "text"
    assert resp.content[0].text == "I am exiting."

def test_exit_tool_streaming_e2e(setup_servers):
    app_port = setup_servers
    client = Anthropic(
        base_url=f"http://localhost:{app_port}",
        api_key="dummy"
    )

    messages = [{"role": "user", "content": "Exit now"}]
    tools = [{
        "name": "dummy",
        "description": "dummy",
        "input_schema": {"type": "object", "properties": {}}
    }]

    print("Sending Streaming Exit request...")
    with client.messages.stream(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=messages,
        tools=tools
    ) as stream:
        final_message = stream.get_final_message()

    print("Got stream final message:", final_message.content)
    # Middleware should have transformed ExitTool call into text
    assert final_message.stop_reason == "end_turn"
    assert len(final_message.content) == 1
    assert final_message.content[0].type == "text"
    assert final_message.content[0].text == "I am exiting."

if __name__ == "__main__":
    # Allow running directly
    sys.exit(pytest.main(["-v", __file__]))
