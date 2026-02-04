import argparse
import threading
import time
import requests
import subprocess
import os
import json
import hashlib
from flask import Flask, request, Response
from deepdiff import DeepDiff
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

app = Flask(__name__)
console = Console()

# Global Configuration & State
TARGET_BASE_URL = "http://localhost:3000"
CURRENT_MODEL_OVERRIDE = None
CURRENT_RECORDING = []
CACHE_DIR = ".model_diff_cache"

def get_cache_key(req_json, model_override):
    """Generate a cache key based on request content (excluding model) and the model override."""
    # We exclude 'model' from the request body because we inject it.
    # But we include the model_override in the key because different models yield different results.
    if req_json and 'model' in req_json:
        req_copy = req_json.copy()
        del req_copy['model']
    else:
        req_copy = req_json

    # Canonicalize
    s = json.dumps(req_copy, sort_keys=True) + f"|{model_override}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def load_from_cache(key):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def save_to_cache(key, data):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, 'w') as f:
        json.dump(data, f)

def parse_sse(text):
    """Parse SSE stream into list of event data objects."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[6:].strip()
            if data == "[DONE]":
                continue
            try:
                events.append(json.loads(data))
            except:
                pass
    return events

def reconstruct_message(events):
    """Reconstruct a final message state from Anthropic SSE events."""
    message = {
        "content": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "stop_reason": None
    }

    for event in events:
        etype = event.get("type")
        if etype == "message_start":
            msg = event.get("message", {})
            message.update({k: v for k, v in msg.items() if k not in ["content", "usage"]})
            if "usage" in msg:
                message["usage"]["input_tokens"] += msg["usage"].get("input_tokens", 0)

        elif etype == "content_block_start":
            idx = event.get("index")
            block = event.get("content_block", {})
            # Ensure content list is big enough
            while len(message["content"]) <= idx:
                message["content"].append({})
            message["content"][idx] = block

        elif etype == "content_block_delta":
            idx = event.get("index")
            delta = event.get("delta", {})
            if idx < len(message["content"]):
                block = message["content"][idx]
                if delta.get("type") == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif delta.get("type") == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                elif delta.get("type") == "input_json_delta":
                    if "input_buffer" not in block:
                        block["input_buffer"] = ""
                    block["input_buffer"] += delta.get("partial_json", "")

        elif etype == "content_block_stop":
            idx = event.get("index")
            if idx < len(message["content"]):
                block = message["content"][idx]
                # Try to parse collected json buffer
                if "input_buffer" in block:
                    try:
                        block["input"] = json.loads(block["input_buffer"])
                    except:
                        block["input_error"] = "Failed to parse JSON"
                        block["raw_input"] = block["input_buffer"]
                    del block["input_buffer"]

        elif etype == "message_delta":
            delta = event.get("delta", {})
            usage = event.get("usage", {})
            if "stop_reason" in delta and delta["stop_reason"]:
                message["stop_reason"] = delta["stop_reason"]
            if "output_tokens" in usage:
                message["usage"]["output_tokens"] = usage["output_tokens"]

    return message

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f"{TARGET_BASE_URL}/{path}"

    headers = {key: value for (key, value) in request.headers if key != 'Host'}

    # Handle body and model override
    data = request.get_data()
    req_json = None

    if request.is_json or headers.get("Content-Type") == "application/json":
        try:
            req_json = request.get_json(silent=True)
            if req_json and CURRENT_MODEL_OVERRIDE and 'model' in req_json:
                # Use override
                req_json['model'] = CURRENT_MODEL_OVERRIDE
                data = json.dumps(req_json).encode('utf-8')
                headers['Content-Length'] = str(len(data))
        except Exception as e:
            pass

    # Check Cache
    cache_key = None
    if req_json and "messages" in path: # Only cache message generations
        cache_key = get_cache_key(req_json, CURRENT_MODEL_OVERRIDE)
        cached = load_from_cache(cache_key)
        if cached:
            console.print("[dim]Cache Hit[/dim]")

            # Record cached interaction
            record_interaction(path, req_json, None, None, cached_response=cached)

            # Replay response
            # Note: The client expects a stream usually. We can fake it or just send JSON if client handles it.
            # But Claude CLI might expect SSE.
            # For simplicity, we'll convert the cached message back to a simple SSE stream or single chunk if possible.
            # Or just return JSON if the client didn't ask for stream?
            # If client asked for stream=True, we should stream.
            if req_json.get("stream"):
                return Response(stream_cached_response(cached), mimetype="text/event-stream")
            else:
                return Response(json.dumps(cached), mimetype="application/json")

    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=data,
            params=request.args,
            stream=True
        )
    except requests.exceptions.ConnectionError:
        return Response("Proxy Error: Connection Refused", status=502)

    captured_response_body = b""

    def generate():
        nonlocal captured_response_body
        for chunk in resp.iter_content(chunk_size=4096):
            captured_response_body += chunk
            yield chunk

        # After stream ends, record & cache
        if req_json and "messages" in path:
            final_resp = record_interaction(path, req_json, captured_response_body, resp.headers)
            if final_resp and cache_key:
                save_to_cache(cache_key, final_resp)

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.headers.items()
               if name.lower() not in excluded_headers]

    return Response(generate(), status=resp.status_code, headers=headers)

def stream_cached_response(message):
    # Simulate an SSE stream from the reconstructed message
    # 1. message_start
    yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': message.get('id', 'cached'), 'role': 'assistant', 'model': message.get('model', 'cached')}})}\n\n"

    # 2. content blocks
    for i, block in enumerate(message.get("content", [])):
        yield f"data: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': block['type']}})}\n\n"

        if block['type'] == 'text':
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'text_delta', 'text': block['text']}})}\n\n"
        elif block['type'] == 'tool_use':
            # Send start with data
            # Actually tool_use block in start contains id, name, input(empty). Delta has input_json.
            # But reconstruction put it all in block.
            # Let's just send the whole thing? No, protocol is specific.
            # Simpler: just send one big delta with the json.
            yield f"data: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}})}\n\n"
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}})}\n\n"

        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

    # 3. message_delta (stop reason)
    yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': message.get('stop_reason')}, 'usage': {'output_tokens': message.get('usage', {}).get('output_tokens', 0)}})}\n\n"
    yield "data: [DONE]\n\n"

def record_interaction(path, req_json, resp_bytes, resp_headers, cached_response=None):
    if cached_response:
        response_data = cached_response
    else:
        is_stream = "text/event-stream" in resp_headers.get("Content-Type", "")
        if is_stream:
            text = resp_bytes.decode('utf-8', errors='replace')
            events = parse_sse(text)
            response_data = reconstruct_message(events)
        else:
            try:
                response_data = json.loads(resp_bytes)
            except:
                response_data = {"raw": resp_bytes.decode('utf-8', errors='replace')}

    CURRENT_RECORDING.append({
        "request": req_json,
        "response": response_data
    })
    return response_data

def run_flask_thread(port):
    app.run(port=port, debug=False, use_reloader=False)

def identify_successful_tool_calls(recording):
    """
    Identify tool calls that resulted in a successful (non-error) output.
    Returns a list of {"tool_call": block, "tool_result": block, "req_idx": int}
    """
    successes = []

    # Map tool_use_id -> (request_index, block)
    tool_uses = {}

    for i, interaction in enumerate(recording):
        resp = interaction["response"]
        if resp and "content" in resp:
            for block in resp["content"]:
                if block.get("type") == "tool_use":
                    tool_uses[block["id"]] = (i, block)

        # Check requests for results
        req = interaction["request"]
        if req and "messages" in req:
            for msg in req["messages"]:
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    for block in msg["content"]:
                        if block.get("type") == "tool_result":
                            tid = block.get("tool_use_id")
                            if tid in tool_uses:
                                use_idx, use_block = tool_uses[tid]
                                is_error = block.get("is_error", False)
                                if not is_error:
                                    successes.append({
                                        "tool_call": use_block,
                                        "tool_result": block,
                                        "req_idx": use_idx
                                    })
    return successes

def main():
    parser = argparse.ArgumentParser(description="Diff Claude runs across models via ant-router")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run (e.g. claude code 'hi')")
    parser.add_argument("--model-a", required=True, help="First model ID")
    parser.add_argument("--model-b", required=True, help="Second model ID")
    parser.add_argument("--proxy-url", default="http://localhost:3000", help="URL of ant-router")
    parser.add_argument("--port", type=int, default=8090, help="Local proxy port")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")

    args = parser.parse_args()

    if args.clear_cache and os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        console.print("[yellow]Cache cleared.[/yellow]")

    global TARGET_BASE_URL, CURRENT_MODEL_OVERRIDE, CURRENT_RECORDING
    TARGET_BASE_URL = args.proxy_url.rstrip("/")

    t = threading.Thread(target=run_flask_thread, args=(args.port,), daemon=True)
    t.start()

    local_proxy_url = f"http://localhost:{args.port}/v1"
    time.sleep(1)

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = local_proxy_url

    # Run A
    console.print(f"[bold blue]Running with Model A: {args.model_a}[/bold blue]")
    CURRENT_MODEL_OVERRIDE = args.model_a
    CURRENT_RECORDING = []
    try:
        subprocess.run(args.command, env=env, check=False)
    except Exception as e:
        console.print(f"[red]Error running command: {e}[/red]")
    recording_a = CURRENT_RECORDING[:]

    # Run B
    console.print(f"\n[bold blue]Running with Model B: {args.model_b}[/bold blue]")
    CURRENT_MODEL_OVERRIDE = args.model_b
    CURRENT_RECORDING = []
    try:
        subprocess.run(args.command, env=env, check=False)
    except Exception as e:
        console.print(f"[red]Error running command: {e}[/red]")
    recording_b = CURRENT_RECORDING[:]

    # Smart Analysis
    console.print("\n[bold]Smart Analysis[/bold]")

    success_a = identify_successful_tool_calls(recording_a)
    success_b = identify_successful_tool_calls(recording_b)

    if success_a and success_b:
        # Compare the first successful tool call pair
        first_a = success_a[0]
        first_b = success_b[0]

        console.print(f"[green]Found successful tool calls for both models.[/green]")
        console.print(f"Model A (Req #{first_a['req_idx']}): {first_a['tool_call']['name']}")
        console.print(f"Model B (Req #{first_b['req_idx']}): {first_b['tool_call']['name']}")

        # Diff inputs
        diff_inputs = DeepDiff(first_a['tool_call'].get('input'), first_b['tool_call'].get('input'), ignore_order=True)
        if diff_inputs:
            console.print("[yellow]Tool Input Differences:[/yellow]")
            console.print(diff_inputs)
        else:
            console.print("[green]Tool Inputs Match[/green]")

    else:
        console.print("[yellow]Could not find aligned successful tool calls in both recordings.[/yellow]")

    # Basic stats
    console.print(f"\nTotal Interactions: A={len(recording_a)}, B={len(recording_b)}")

    # Final Output Comparison (Last interaction)
    if recording_a and recording_b:
        last_a = recording_a[-1]["response"]
        last_b = recording_b[-1]["response"]

        text_a = "".join([b["text"] for b in last_a.get("content", []) if b["type"] == "text"])
        text_b = "".join([b["text"] for b in last_b.get("content", []) if b["type"] == "text"])

        console.print("\n[bold]Final Response Comparison:[/bold]")
        console.print(Panel(text_a, title=f"Model A ({args.model_a})"))
        console.print(Panel(text_b, title=f"Model B ({args.model_b})"))

if __name__ == "__main__":
    main()
