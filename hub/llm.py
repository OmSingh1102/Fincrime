"""Claude through Straive's LLM Foundry gateway, for the hub's on-demand
agents. The same recipe as the console's model assessment
(fincrime/engines/ai_review.py): the official Anthropic SDK pointed at the
gateway's /anthropic route with the gateway token as a bearer token,
adaptive thinking, medium effort and a JSON schema for the answer.

Configuration, read once at import (documented in .env.example):
  LLM_FOUNDRY_KEY   the gateway token; unset = the Claude agents are off
  LLM_FOUNDRY_URL   the gateway (default https://llmfoundry.straivedemo.com)
  LLM_MODEL         the Claude model id (default claude-opus-4-8, the strongest
                    the gateway enables)

call() is the one function that talks to the gateway; the tests stand in
for it.
"""
import json
import os
import time
from urllib.parse import urlsplit

from fincrime.engines.errors import WorkflowError

GATEWAY = os.environ.get("LLM_FOUNDRY_URL", "https://llmfoundry.straivedemo.com").rstrip("/")
KEY = os.environ.get("LLM_FOUNDRY_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "claude-opus-4-8")
TIMEOUT = 90.0

_state = {"client": None, "calls": 0, "last_error": "", "seconds": []}


def configured():
    return bool(KEY)


def switched_on(settings):
    return bool((settings or {}).get("claude", True))


def why_not(settings):
    """Why the buttons are off, or "" when a call can run."""
    if not configured():
        return "Claude is not configured: set LLM_FOUNDRY_KEY (see .env.example)."
    if not switched_on(settings):
        return "The Claude agents are switched off on the Agents page."
    return ""


def status(settings=None):
    why = why_not(settings)
    secs = _state["seconds"]
    return {"configured": configured(), "on": switched_on(settings), "ready": not why, "why": why, "model": MODEL,
            "gateway": urlsplit(GATEWAY).netloc, "calls": _state["calls"], "last_error": _state["last_error"],
            "avg_seconds": round(sum(secs) / len(secs), 1) if secs else 0.0}


def _client():
    if _state["client"] is None:
        import anthropic
        _state["client"] = anthropic.Anthropic(base_url=GATEWAY + "/anthropic", auth_token=KEY, api_key=None,
                                               timeout=TIMEOUT, max_retries=1)
    return _state["client"]


def call(system, text, schema, max_tokens=3000):
    """One request to the gateway. Returns (parsed JSON object, usage,
    seconds). Every failure is a WorkflowError the screen can show."""
    import anthropic
    started = time.monotonic()
    try:
        response = _client().messages.create(
            model=MODEL, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": text}],
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.AuthenticationError:
        raise WorkflowError("The model gateway rejected the key (LLM_FOUNDRY_KEY).")
    except anthropic.APIStatusError as exc:
        raise WorkflowError(f"The model gateway answered {exc.status_code}: {exc.message}")
    except anthropic.APIConnectionError:
        raise WorkflowError("Could not reach the model gateway.")
    if response.stop_reason == "refusal":
        raise WorkflowError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        raise WorkflowError("The model's answer was cut short.")
    answer = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(answer)
    except ValueError:
        raise WorkflowError("The model's answer was not the JSON expected.")
    u = response.usage
    usage = {"input": u.input_tokens, "output": u.output_tokens, "cached": getattr(u, "cache_read_input_tokens", 0) or 0}
    return data, usage, round(time.monotonic() - started, 1)


def record(seconds, error=""):
    """Bookkeeping for the status chip: a call completed, or failed."""
    if error:
        _state["last_error"] = error
        return
    _state["calls"] += 1
    _state["last_error"] = ""
    _state["seconds"].append(seconds)
