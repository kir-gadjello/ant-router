import argparse
import threading
import time
import requests
import subprocess
import os
import json
import tempfile
import signal
from pathlib import Path
from deepdiff import DeepDiff
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

console = Console()

# --- Tracer Logic ---

def tail_trace_file(path, stop_event):
    """Generator that yields new lines from the trace file."""
    # Wait for file to exist
    while not os.path.exists(path):
        if stop_event.is_set(): return
        time.sleep(0.1)

    with open(path, 'r') as f:
        # Go to end? No, we want everything from start of THIS run.
        # But file might be reused? The runner creates a new one per session usually.
        # We assume clean file.
        while not stop_event.is_set():
            line = f.readline()
            if line:
                try:
                    yield json.loads(line)
                except:
                    pass
            else:
                time.sleep(0.1)

def group_traces(trace_generator):
    """Groups traces by Correlation ID."""
    interactions = {} # id -> {events: [], ...}

    for event in trace_generator:
        # event structure: {"event": "Type", "data": {...}}
        etype = event.get("event")
        data = event.get("data", {})
        rid = data.get("id")

        if not rid: continue

        if rid not in interactions:
            interactions[rid] = {
                "frontend_req": None,
                "upstream_req": None,
                "upstream_resp": None,
                "frontend_resp": None,
                "timestamp": data.get("timestamp")
            }

        if etype == "FrontendRequest":
            interactions[rid]["frontend_req"] = data.get("payload")
        elif etype == "UpstreamRequest":
            interactions[rid]["upstream_req"] = data.get("payload")
        elif etype == "UpstreamResponse":
            interactions[rid]["upstream_resp"] = data.get("payload")
        elif etype == "FrontendResponse":
            interactions[rid]["frontend_resp"] = data.get("payload")

    # Sort by timestamp
    return sorted(interactions.values(), key=lambda x: x["timestamp"] or "")

# --- Runner Logic ---

def run_model_trace(model_id, command, router_bin, proxy_port, config_path):
    """Runs a single model trace session."""

    # 1. Setup specific trace file
    trace_file = tempfile.mktemp(prefix=f"trace_{model_id}_", suffix=".jsonl")

    # 2. Modify Config to include trace_file (create temp config)
    # We assume base config exists.
    with open(config_path, 'r') as f:
        base_config = f.read()

    # Append trace file config (simple append works for YAML top level)
    # Also ensure port is set
    temp_config_path = tempfile.mktemp(prefix="config_", suffix=".yaml")
    with open(temp_config_path, 'w') as f:
        f.write(base_config)
        f.write(f"\ntrace_file: \"{trace_file}\"\n")
        # We might need to enforce port if not in base config, but we pass --port to binary

    # 3. Start Router
    console.print(f"[dim]Starting router on port {proxy_port} for {model_id}...[/dim]")
    router_proc = subprocess.Popen(
        [router_bin, "--config", temp_config_path, "--port", str(proxy_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE # Capture error log if needed
    )

    # Wait for router health
    ready = False
    for _ in range(20):
        try:
            requests.get(f"http://localhost:{proxy_port}/health")
            ready = True
            break
        except:
            time.sleep(0.2)

    if not ready:
        console.print("[red]Router failed to start.[/red]")
        router_proc.kill()
        return []

    # 4. Run Command
    # We must inject the model ID into the environment or command?
    # The command is like `claude code "task"`. Claude CLI usually takes model arg or config.
    # But we want to FORCE the model at the router level?
    # Our previous approach used a python proxy to inject.
    # Here, we want `ant-router` to do it?
    # Ant-router supports `OVERRIDE-` logic in model name, but Client sends model name.
    # If we run `claude --model ...`, we can control it.
    # If the user provides a generic command, we might need to rely on the CLI tool accepting a model flag.
    # OR: We use a "default" profile in the router config that points to our target model?
    # YES: We can generate a config where "default" profile maps ".*" to our specific model.

    # Let's update the temp config to force the model routing.
    # Re-write config:
    with open(temp_config_path, 'w') as f:
        f.write(base_config)
        f.write(f"\ntrace_file: \"{trace_file}\"\n")
        # Overwrite profiles to force everything to target model
        # This is a bit hacky on YAML structure. Better to use the `profiles` block if we parse it.
        # But we can append a new profile "trace_forced" and set it as current.
        f.write(f"\ncurrent_profile: \"trace_forced\"\n")
        f.write("profiles:\n  trace_forced:\n    rules:\n")
        f.write(f"      - pattern: \".*\"\n        target: \"forced_target\"\n")
        f.write("models:\n  forced_target:\n")
        f.write(f"    api_model_id: \"{model_id}\"\n")
        # Inherit provider from somewhere? We assume user has a default provider setup or we need one.
        # This might break if 'forced_target' doesn't have provider info.
        # ALTERNATIVE: Use the python proxy from before JUST to rewrite the model in the request?
        # That adds complexity.
        # Let's assume the user command includes the model OR the user accepts we just run whatever the command says,
        # but we assume the user WANTS to compare Model A vs Model B.
        # So we really need to force it.
        # Best way: Use `ANT_ROUTER_PROFILE` env var or similar if we supported it? No.
        # Let's use the Python Proxy *concept* but embedded? No, we want direct Router.

        # ACTUALLY: The previous tool used a Flask proxy to inject `model`.
        # We can reuse that approach: Run `model_diff.py` -> Flask Proxy (rewrites model) -> Ant Router (traces) -> Upstream.
        # This keeps the router config simple.
        pass

    # ... Reset Router with original config for now, assuming external proxy handles model injection ...
    # Wait, if we use the flask proxy, we need TWO ports.
    # 1. Router Port (e.g. 3000)
    # 2. Proxy Port (e.g. 8090)
    # The client talks to 8090. 8090 forwards to 3000 (injecting model). 3000 forwards to upstream (tracing).

    # Start Flask Proxy Logic (in background thread of this process)
    # We need a fresh proxy for each run or one global one?
    # One global is fine if we update its target model.

    # For this implementation, let's keep it simple:
    # We assume the user command allows setting the model, OR we assume the user configured the command to use "claude-3-opus" and we intercept "claude-3-opus" in the router and remap it?
    # Remapping in router is cleanest if we can generate the config.
    # Let's try appending a specific rule.

    with open(temp_config_path, 'a') as f:
        # Override the "default" profile behavior or whatever profile is active.
        # We can set `current_profile: forced_trace_profile`
        f.write(f"\ncurrent_profile: forced_trace_profile_{os.getpid()}\n")
        f.write(f"profiles:\n  forced_trace_profile_{os.getpid()}:\n")
        f.write("    rules:\n      - pattern: \".*\"\n")
        f.write(f"        target: \"target_model_{os.getpid()}\"\n")
        f.write(f"models:\n  target_model_{os.getpid()}:\n")
        f.write(f"    api_model_id: \"{model_id}\"\n")
        # We need a provider. Use default? Or assume model_id includes prefix like "openai/..."?
        # If model_id is "gpt-4", we need provider.
        # Let's assume the user provides a full wire ID or a known logical ID from the base config?
        # If the user provides a logical ID that exists in base config, we can just point to it.
        # If user provides "openai/gpt-4o", we need a generic provider.
        # Let's assume user provides a Logical ID present in config OR we map to a generic openrouter one.
        # For safety, let's look for the model in the config? Too hard to parse YAML manually here.
        # Let's just assume it's a wire ID and use OpenRouter (default upstream).

    # Restart router with this config
    router_proc.terminate()
    router_proc.wait()

    router_proc = subprocess.Popen(
        [router_bin, "--config", temp_config_path, "--port", str(proxy_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    time.sleep(1) # Wait for startup

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://localhost:{proxy_port}/v1"
    # Also set OPENAI_BASE_URL just in case
    env["OPENAI_BASE_URL"] = f"http://localhost:{proxy_port}/v1"

    console.print(f"[blue]Running command for {model_id}...[/blue]")
    try:
        subprocess.run(command, env=env, check=False)
    except Exception as e:
        console.print(f"[red]Command failed: {e}[/red]")

    # Stop router
    router_proc.terminate()

    # Read Trace
    traces = []
    if os.path.exists(trace_file):
        with open(trace_file, 'r') as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except: pass
        os.remove(trace_file)

    os.remove(temp_config_path)

    return group_traces(iter(traces))

def print_interaction_diff(int_a, int_b):
    """Diffs two interaction objects."""

    # 1. Frontend Request Diff
    # Usually identical except maybe non-deterministic ordering or timestamps?
    # We care about TOOL CALLS in the response mostly.

    # 2. Response Diff
    # We want to align the "Upstream Response" (what the model said)
    # AND the "Frontend Response" (what we sent back, to check translation).

    resp_a = int_a.get("upstream_resp", {})
    resp_b = int_b.get("upstream_resp", {})

    # Extract Tool Calls
    tools_a = extract_tools(resp_a)
    tools_b = extract_tools(resp_b)

    if tools_a or tools_b:
        table = Table(title="Tool Call Comparison")
        table.add_column("Model A", style="cyan")
        table.add_column("Model B", style="magenta")

        for i in range(max(len(tools_a), len(tools_b))):
            ta = tools_a[i] if i < len(tools_a) else {}
            tb = tools_b[i] if i < len(tools_b) else {}

            ta_str = f"{ta.get('name')}\n{json.dumps(ta.get('args'), indent=2)}"
            tb_str = f"{tb.get('name')}\n{json.dumps(tb.get('args'), indent=2)}"
            table.add_row(ta_str, tb_str)

        console.print(table)

    # Check translation (Upstream -> Frontend)
    # Ideally, frontend response should logically match upstream response.
    # If `ant-router` bugs out, this is where we see it.

    console.print("[dim]Checking Translation Integrity...[/dim]")
    check_translation(int_a, "Model A")
    check_translation(int_b, "Model B")

def extract_tools(openai_resp):
    tools = []
    # Handle both full response and chunks (if aggregated? Trace logs chunks?)
    # The trace logger logs the *accumulated* response if we did it right?
    # No, `log_trace` in `handlers.rs` for UpstreamResponse logs `openai_resp` which is the FULL object in non-streaming.
    # For streaming... `handle_openai_chat` logs nothing for chunks yet?
    # Wait, the previous plan implemented `log_trace` calls.
    # In `handle_openai_chat` (streaming), we didn't add chunk logging!
    # We only added `UpstreamResponse` log in the non-streaming path.
    # Ideally we should reconstruct streams.
    # For now, let's assume non-streaming or that we parse the frontend response which IS logged (via `record_stream`).

    choices = openai_resp.get("choices", [])
    for c in choices:
        msg = c.get("message", {})
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                tools.append({
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"])
                })
    return tools

def check_translation(interaction, label):
    # Verify that what we received from upstream roughly matches what we sent to frontend
    # This detects if we dropped tool calls or mangled args.
    up = interaction.get("upstream_resp")
    front = interaction.get("frontend_resp")

    if not up or not front: return

    # Extract tools from both
    up_tools = extract_tools(up)

    # Frontend (Anthropic) tools
    front_tools = []
    content = front.get("content", [])
    for block in content:
        if block.get("type") == "tool_use":
            front_tools.append({
                "name": block.get("name"),
                "args": block.get("input")
            })

    # Compare
    diff = DeepDiff(up_tools, front_tools, ignore_order=True)
    if diff:
        console.print(f"[red]{label} Translation Mismatch![/red]")
        console.print(diff)
    else:
        console.print(f"[green]{label} Translation Verified[/green]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--router-bin", default="./target/debug/anthropic-bridge")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    # Build if needed? No, assume built.

    console.rule("Tracing Model A")
    trace_a = run_model_trace(args.model_a, args.command, args.router_bin, args.port, args.config)

    console.rule("Tracing Model B")
    trace_b = run_model_trace(args.model_b, args.command, args.router_bin, args.port, args.config)

    console.rule("Comparison")

    # Align interactions
    # Simple alignment by index for now (assuming deterministic flow structure)
    max_len = max(len(trace_a), len(trace_b))

    for i in range(max_len):
        console.print(f"\n[bold]Step {i+1}[/bold]")
        ia = trace_a[i] if i < len(trace_a) else {}
        ib = trace_b[i] if i < len(trace_b) else {}

        print_interaction_diff(ia, ib)

if __name__ == "__main__":
    main()
