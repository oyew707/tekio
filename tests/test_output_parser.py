from types import SimpleNamespace

from app.agent.output_parser import attach_fallback_tool_calls, extract_tool_calls


def test_extract_tool_calls_from_xml_block():
    text = '<tool_call>{"name": "computer_use", "arguments": {"action": "click", "x": 500, "y": 500}}</tool_call>'
    calls = extract_tool_calls(text)

    assert len(calls) == 1
    assert calls[0]["name"] == "computer_use"
    assert calls[0]["args"]["action"] == "click"


def test_attach_fallback_tool_calls_populates_message():
    msg = SimpleNamespace(content='<tool_call>{"name": "computer_use", "arguments": {"action": "goto", "url": "https://example.com"}}</tool_call>', tool_calls=[])

    updated = attach_fallback_tool_calls(msg)

    assert updated.tool_calls[0]["name"] == "computer_use"
    assert updated.tool_calls[0]["args"]["url"] == "https://example.com"
