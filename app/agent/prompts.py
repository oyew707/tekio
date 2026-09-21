"""Prompt templates for the browser-use agent."""

from __future__ import annotations

BASE_SYSTEM_PROMPT = """You are a browser automation assistant.
Use only the `computer_use` tool for browser interaction.
Coordinates are normalized to a 1000x1000 plane.
When tool use is needed, emit:
<tool_call>{"name": "computer_use", "arguments": {...}}</tool_call>
"""


def build_system_prompt(rag_tips: list[str] | None = None) -> str:
    rag_tips = rag_tips or []
    if not rag_tips:
        return BASE_SYSTEM_PROMPT

    formatted = "\n".join(f"- {tip}" for tip in rag_tips)
    return (
        f"{BASE_SYSTEM_PROMPT}\n"
        "Retrieved prior guidance:\n"
        f"{formatted}\n"
    )
