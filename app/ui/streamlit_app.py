"""Streamlit UI for chat + browser snapshot view."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st

from app.agent.graph import BrowserUseAgent
from app.browser.action_executor import ActionExecutor
from app.browser.playwright_session import PlaywrightSession
from app.rag.store import TipStore
from app.tracing.queue_publisher import TrajectoryQueuePublisher

LATEST_SCREENSHOT = Path("artifacts/latest.png")


def _ensure_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    st.set_page_config(layout="wide", page_title="tekio browser-use")
    _ensure_state()

    left, right = st.columns(2)
    task = left.chat_input("Describe the browser task")

    with left:
        st.subheader("Chat")
        for role, message in st.session_state.history:
            with st.chat_message(role):
                st.write(message)

    with right:
        st.subheader("Browser")
        if LATEST_SCREENSHOT.exists():
            st.image(str(LATEST_SCREENSHOT), use_container_width=True)
        else:
            st.info("No screenshot yet.")

    if not task:
        return

    st.session_state.history.append(("user", task))

    async def _run() -> str:
        session = PlaywrightSession(headless=os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true")
        await session.start()
        try:
            executor = ActionExecutor(session)
            rag_store = TipStore() if os.getenv("PGVECTOR_URL") else None
            agent = BrowserUseAgent(executor, rag_store=rag_store)
            result = await agent.run(task)
            trace_id = os.getenv("LANGCHAIN_RUN_ID", "unknown")
            if os.getenv("RABBITMQ_URL"):
                TrajectoryQueuePublisher().publish(trace_id=trace_id, task=task, status="completed")
            return str(result.get("final", ""))
        finally:
            await session.stop()

    response = asyncio.run(_run())
    st.session_state.history.append(("assistant", response))
    st.rerun()


if __name__ == "__main__":
    main()
