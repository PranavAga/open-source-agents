"""Guardrails: path sandbox, secret hygiene, and output-contract checks."""
import fnmatch
import json
import os
import re
from pathlib import Path

from google.adk.models.llm_response import LlmResponse
from google.genai import types

DIMENSIONS = [
    "Usefulness & clarity",
    "Ease of adoption",
    "Maintainability & contributor friction",
    "Distribution & scale",
    "Friction red flags",
]
SECRET_GLOBS = [".env*", "*.pem", "*.key", "id_rsa*"]
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}")
SIGNAL_RE = re.compile(r"\b(?:missing|present|flag|adoption):[\w=./-]+")


def review_root() -> Path:
    return Path(os.environ.get("REVIEW_ROOT", ".")).resolve()


def safe_path(path: str) -> Path:
    """Resolve `path` (following symlinks) and require it to stay under REVIEW_ROOT."""
    root = review_root()
    p = Path(path)
    resolved = (p if p.is_absolute() else root / p).resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"path escapes REVIEW_ROOT: {path}")
    return resolved


def is_secret_file(path: Path) -> bool:
    name = path.name
    if name == ".env.example":
        return False
    return any(fnmatch.fnmatch(name, g) for g in SECRET_GLOBS)


def redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def before_model_callback(callback_context, llm_request):
    """Cap LLM calls per invocation (MAX_LLM_CALLS), since `adk run`/`adk web` accept no RunConfig."""
    calls = callback_context.state.get("temp:llm_calls", 0) + 1
    callback_context.state["temp:llm_calls"] = calls
    if calls > int(os.environ.get("MAX_LLM_CALLS", "15")):
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(
            text="Stopped: LLM call limit reached (MAX_LLM_CALLS). Re-run with a higher limit if needed.")]))
    return None


def before_tool_callback(tool, args, tool_context):
    """Second enforcement of the sandbox, in case the MCP server is bypassed."""
    path = args.get("path")
    if path is None:
        return None
    try:
        resolved = safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    if tool.name == "read_file" and is_secret_file(resolved):
        return {"error": f"refusing to read secret file: {resolved.name}"}
    return None


def scan_scores(tool_response) -> dict | None:
    """Per-dimension scores from a `scan_repo` MCP result (structured content, else the JSON text block)."""
    if not isinstance(tool_response, dict):
        return None
    data = tool_response.get("structuredContent")
    if isinstance(data, dict) and "result" in data and "dimensions" not in data:
        data = data["result"]
    if not isinstance(data, dict):
        try:
            data = json.loads(tool_response["content"][0]["text"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    try:
        return {d["name"]: int(d["score"]) for d in data["dimensions"]}
    except (KeyError, TypeError, ValueError):
        return None


def after_tool_callback(tool, args, tool_context, tool_response):
    """Remember the deterministic scores so the final report can be held to them."""
    if tool.name == "scan_repo" and (scores := scan_scores(tool_response)):
        tool_context.state["temp:scores"] = scores
    return None


def enforce_scores(text: str, scores: dict) -> tuple[str, list[str]]:
    """Rewrite any `<dimension> ... N/5` heading whose N differs from the scan."""
    fixed = []
    for name, score in scores.items():
        pat = re.compile(rf"^(#{{1,6}}[^\n]*?{re.escape(name)}[^\n]*?)(\d+)(\s*/\s*5)", re.I | re.M)
        m = pat.search(text)
        if m and int(m.group(2)) != score:
            fixed.append(f"{name} {m.group(2)}→{score}")
            text = pat.sub(lambda m: f"{m.group(1)}{score}{m.group(3)}", text, count=1)
    return text, fixed


def after_model_callback(callback_context, llm_response):
    """Redact secrets, hold reported scores to `scan_repo`, and warn when the report breaks the output contract."""
    content = llm_response.content
    if llm_response.partial or not content or not content.parts:
        return None
    if any(part.function_call for part in content.parts):
        return None
    text = "".join(part.text or "" for part in content.parts if not part.thought)
    if not text.strip():
        return None
    text = redact(text)
    lowered = text.lower()
    if not re.search(r"/\s*5\b", text) and "top 5 fixes" not in lowered:
        # Not a report (e.g. asking the user for a path): redact only, no contract check.
        llm_response.content = types.Content(role="model", parts=[types.Part(text=text)])
        return llm_response
    scores = callback_context.state.get("temp:scores") if callback_context else None
    if scores:
        text, fixed = enforce_scores(text, scores)
        if fixed:
            text += "\n\n> ℹ️ Scores corrected to match `scan_repo`: " + ", ".join(fixed) + "."
            lowered = text.lower()
    problems = []
    missing = [d for d in DIMENSIONS if d.lower() not in lowered]
    if missing:
        problems.append("missing dimension sections: " + ", ".join(missing))
    if not SIGNAL_RE.search(text):
        problems.append("no cited signal ids (e.g. missing:license)")
    if problems:
        text += "\n\n> ⚠️ Output check failed: " + "; ".join(problems) + "."
    llm_response.content = types.Content(role="model", parts=[types.Part(text=text)])
    return llm_response
