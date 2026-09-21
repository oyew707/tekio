"""Fallback parser for models that emit XML-wrapped tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract OpenAI-style tool calls from <tool_call>{...}</tool_call> blocks."""
    matches = TOOL_CALL_RE.findall(text or "")
    tool_calls: list[dict[str, Any]] = []

    for idx, raw_json in enumerate(matches):
        payload = json.loads(raw_json)
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        tool_calls.append(
            {
                "id": payload.get("id", f"fallback_call_{idx}"),
                "type": "tool_call",
                "name": name,
                "args": arguments,
            }
        )

    return tool_calls


def attach_fallback_tool_calls(ai_message: Any) -> Any:
    """Populate ai_message.tool_calls if empty but XML-wrapped payload exists."""
    content = getattr(ai_message, "content", "")
    if getattr(ai_message, "tool_calls", None):
        return ai_message

    tool_calls = extract_tool_calls(content if isinstance(content, str) else str(content))
    if tool_calls:
        ai_message.tool_calls = tool_calls

    return ai_message
