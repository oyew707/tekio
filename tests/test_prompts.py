from app.agent.prompts import build_system_prompt


def test_build_system_prompt_includes_rag_guidance_block():
    prompt = build_system_prompt(["Always wait for page load", "Prefer keyboard shortcuts when possible"])

    assert "Retrieved prior guidance" in prompt
    assert "- Always wait for page load" in prompt
    assert "- Prefer keyboard shortcuts when possible" in prompt
