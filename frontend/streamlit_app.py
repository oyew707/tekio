"""
-------------------------------------------------------
Streamlit UI for chat + browser snapshot view.
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
from dotenv import load_dotenv
import os
import threading
from mcp.shared.exceptions import MCPError
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx
import asyncio
from agent.graph import build_graph, AVAILABLE_AGENTIC_MODELS
from playwright.async_api import async_playwright
from langchain.mcp import MCPAdapter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from workers.queue_publisher import TrajectoryQueuePublisher
from utils.logger import get_logger

# Constants
load_dotenv()
logger = get_logger(__name__, "info")
SCREENSHOT_PLACEHOLDER = os.path.join(os.getcwd(), "frontend/dino_game.webp")
CDP_ENDPOINT = os.getenv("CDP_ENDPOINT", "http://localhost:9222")
QUEUE_ENDPOINT = os.getenv("RABBITMQ_URL")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "trajectory_ready")
STREAMLIT_STYLE = """
<style>
    /* Hide the streamlit deploy button */
    .stDeployButton {
        visibility: hidden;
    }
    section[data-testid="stSidebar"] {
        width: 360px !important;
    }
    /* Make the chat input stick to the bottom */
    .stChatInputContainer {
        position: sticky;
        bottom: 0;
        background: white;
        z-index: 999;
    }
</style>
"""
DEFAULT_STATE = {
    "history": StreamlitChatMessageHistory(key="chat_history"),
    "max_tokens": 8192,
    "last_error": None,
    "model": "browser-use-9b",
    "system_prompt": "",  # Additional Prompts
    "frame": None,
    "chat_disabled": False,
    "watching": False,
}
CONFIG = {
    "mcpServers": {
        "playwright": {
            "url": "http://playwright-mcp:8931/mcp",
            "sse_read_timeout": 1800.0,
            "timeout": 30.0,
        }
    }
}
MAX_RETRIES = 3


def setup_state():
    """
    -------------------------------------------------------
    Initialize/reload session state varaibles
    -------------------------------------------------------
    """
    logger.info("Initializing session state")
    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v


async def _find_agent_page(browser):
    """
    -------------------------------------------------------
    Best-effort guess at which page is being driven by the MCP browser agent (skips blank tabs).
    -------------------------------------------------------
    Parameters:
       browser -
    Returns:
       page -
    -------------------------------------------------------
    """
    for context in browser.contexts:
        for page in context.pages:
            if page.url not in ("about:blank", ""):
                logger.debug(f"Found non-blank page: {page.url}")
                return page
    logger.debug("No active non-blank page found; falling back to default page")
    context = browser.contexts[0] if browser.contexts else await browser.new_context()
    return context.pages[0] if context.pages else await context.new_page()


async def _watch_loop():
    """
    -------------------------------------------------------
    Connects to the agent's browser over CDP and continuously
    grabs screenshots into session_state.frame. Runs in its
    own background thread/event loop, fully decoupled from
    whatever the agent itself is doing via the MCP server.
    -------------------------------------------------------
    """
    logger.info(f"Starting CDP watch loop on {CDP_ENDPOINT}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(CDP_ENDPOINT)
            st.session_state.watching = True
            logger.debug("Browser connected via CDP")
            while st.session_state.watching:
                try:
                    page = await _find_agent_page(browser)
                    st.session_state.frame = await page.screenshot(
                        type="jpeg", quality=60
                    )
                    logger.debug("Captured browser frame")
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error during screenshot iteration: {e}")
                    break
            await browser.close()
            logger.debug("Browser closed")
    finally:
        logger.info("Stopping CDP watch loop")
        st.session_state.watching = False


def start_watching():
    """
    -------------------------------------------------------
    Starts the watch loop
    -------------------------------------------------------
    """
    if st.session_state.watching:
        logger.warning("Start watching called while already watching")
        return

    def thread_wrapper():
        asyncio.run(_watch_loop())

    thread = threading.Thread(target=thread_wrapper, daemon=True)
    add_script_run_ctx(thread)
    thread.start()


def stop_watching():
    """
    -------------------------------------------------------
    Stops the watch loop
    -------------------------------------------------------
    """
    logger.info("Stopping background watcher")
    st.session_state.watching = False


def main():
    """
    -------------------------------------------------------
    Main Entrypoint
    -------------------------------------------------------
    """

    async def run(agent_state):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with MCPAdapter(CONFIG) as adapter:
                    # Agent
                    logger.info("Building agent graph")
                    agent = await build_graph(adapter=adapter)

                    logger.debug(
                        f"Running agent stream with model: {st.session_state.model}"
                    )
                    final_state = await agent.ainvoke(agent_state)
                return final_state
            except MCPError as e:
                if e.code == -32600 or "Session terminated" in str(e):
                    logger.error(
                        f"MCP Session expired or terminated (Attempt {attempt}/{MAX_RETRIES}). Reinitializing client..."
                    )
                    await asyncio.sleep(2)
                    continue
                raise e
        raise RuntimeError("MCP Session could not be stabilized after max retries.")

    # App
    st.set_page_config(layout="wide", page_title="tekio browser-use", page_icon="☸")
    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    setup_state()

    with st.sidebar:
        st.header("Configuration")

        # Max tokens
        st.number_input(
            "Max Output Tokens",
            min_value=1024,
            max_value=32768,
            step=1024,
            key="max_tokens",
        )

        st.selectbox("Model", options=AVAILABLE_AGENTIC_MODELS, index=0, key="model")

        # System prompt
        st.text_area(
            "Additional System Prompt",
            value=st.session_state.system_prompt,
            key="system_prompt",
            help="Add custom instructions for the browser agent",
        )

        st.divider()
        st.caption("Browser live view")

        c1, c2 = st.columns(2)
        if c1.button(
            "Start watching",
            disabled=st.session_state.watching,
            width="stretch",
        ):
            start_watching()
        if c2.button("Stop", disabled=not st.session_state.watching, width="stretch"):
            stop_watching()

        if st.session_state.watching:
            st.caption("🟢 Watching")
        else:
            st.caption("⚪ Not watching")

    left, right = st.columns(2)
    with left:
        st.subheader("Conversation")

        for msg in st.session_state.history.messages:
            with st.chat_message(msg.type):
                st.write(msg.content)

        # Show status when chat is disabled
        if st.session_state.chat_disabled:
            st.info("Agent is currently processing your request. Please wait...")

        # Simple chat input with disabled state
        if prompt := st.chat_input(
            "Ask Tekio to browse the web...",
            disabled=st.session_state.get("chat_disabled", False),
        ):
            logger.info(f"Received user prompt: {prompt}")
            agent_state = {
                "user_input": prompt,
                "router_decision": None,
                "model": st.session_state.model,
                "user_mesg": st.session_state.history.messages,
            }
            # Display user message and add to history
            st.chat_message("user").write(prompt)
            st.session_state.history.add_user_message(prompt)
            st.session_state.chat_disabled = True

            # Start watching the browser
            start_watching()

            # Process the prompt
            try:
                final_state = asyncio.run(run(agent_state=agent_state))
                response_text = final_state.get("message")[-1].content
                logger.info(f"Agent router decision: {response_text}")
                st.chat_message("assistant").write(response_text)
                st.session_state.history.add_ai_message(response_text)
                if QUEUE_ENDPOINT and final_state.get("trace_id") is not None:
                    logger.debug(
                        f"Publishing trajectory for trace_id: {final_state.get('trace_id')}"
                    )
                    TrajectoryQueuePublisher(queue_name=QUEUE_NAME).publish(
                        trace_id=final_state.get("trace_id"),
                        task=prompt,
                        status="completed",
                    )
                    logger.info("Successfully published trajectory")
            except Exception as e:
                logger.error(f"Error during agent execution: {e}", exc_info=True)
                st.error(f"An error occurred: {e}")
            finally:
                st.session_state.chat_disabled = False
                st.rerun()

    with right:
        st.subheader("Browser")

        @st.fragment(run_every=0.15)
        def live_view():
            if st.session_state.frame:
                st.image(st.session_state.frame, width="stretch")
            elif st.session_state.watching:
                st.write("Connecting...")
            else:
                st.info("Not watching yet. Start it from the sidebar.")

        live_view()


if __name__ == "__main__":
    main()
