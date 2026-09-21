"""LangChain agent loop using a single `computer_use` tool contract."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.agent.output_parser import attach_fallback_tool_calls
from app.agent.prompts import build_system_prompt


class BrowserUseAgent:
    """Minimal agent loop that executes model-emitted computer_use actions."""

    def __init__(self, action_executor: Any, rag_store: Any | None = None, max_steps: int = 10) -> None:
        self.action_executor = action_executor
        self.rag_store = rag_store
        self.max_steps = max_steps
        self.llm = ChatOpenAI(
            base_url=os.environ["OPENAI_API_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_MODEL_NAME", "fara-1.5-XXb"),
        )

    async def run(self, task: str) -> dict[str, Any]:
        rag_hits = self.rag_store.query(task, k=3) if self.rag_store else []
        rag_tips = [item["content"] for item in rag_hits]

        messages: list[Any] = [
            SystemMessage(content=build_system_prompt(rag_tips)),
            HumanMessage(content=task),
        ]

        for _ in range(self.max_steps):
            ai_message = self.llm.invoke(messages)
            ai_message = attach_fallback_tool_calls(ai_message)
            messages.append(ai_message)

            tool_calls = getattr(ai_message, "tool_calls", []) or []
            if not tool_calls:
                return {"final": ai_message.content, "messages": messages}

            for call in tool_calls:
                if call.get("name") != "computer_use":
                    continue
                result = await self.action_executor.execute(call.get("args", {}))
                messages.append(
                    ToolMessage(
                        tool_call_id=call.get("id", "computer_use_call"),
                        name="computer_use",
                        content=result,
                    )
                )

        return {"final": "Max steps reached.", "messages": messages}
