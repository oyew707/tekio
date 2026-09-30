"""
-------------------------------------------------------
Graph Nodes (Actor and Reviewer)
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import os
import uuid
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langsmith.run_helpers import get_current_run_tree
from typing import List, Literal
from agent.prompt import SYSTEM_PROMPT, ROUTER_PROMPT
from .state import AgentState
from rag.store import TipStore
from utils.logger import get_logger
from utils.agent_utils import capture_and_build_screen_message

# Constants
logger = get_logger(__name__, "debug")
rag_store: TipStore | None = TipStore()
initial_tool_call = {
    "name": "visit_url",
    "args": {"url": "https://duckduckgo.com"},
    "id": f"call_{uuid.uuid4().hex[:12]}",
}
observation_id = f"message_{uuid.uuid4().hex[:12]}"


class RouterOutput(BaseModel):
    """
    -------------------------------------------------------
    Router on whether Task is complete or cant progress
    or Agent needs to continue
    -------------------------------------------------------
    Parameters:
        status - to determine whether to go back to the agent or user
        message - message to Agent to continue its workflow or to user if complete (str)
    Returns:
        RouterOutput - Structured output of next steps (RouterOutput)
    -------------------------------------------------------
    """

    message: str = Field(
        description="message to Agent to continue its workflow or to user if complete"
    )
    status: Literal["CONTINUE", "SUCCEEDED", "FAILED", "AWAITING_INPUT"]


async def agent_node(state: AgentState, tools: List, playright_tools: List):
    """
    -------------------------------------------------------
    Runs the entire Agent
    -------------------------------------------------------
    Parameters
        state (AgentState) - The current agent state containing user_input, router_decision,
            messages, iterations, trace_id, and model configuration.
        tools (List) - List of agent tools available for tool calling (e.g., browser actions).
        playright_tools (List) - List of Playwright tools for browser interaction (e.g., browser_evaluate,
            browser_find, browser_take_screenshot).
    Returns:
        dict - A dictionary containing updates to state:
            - messages (List[BaseMessage]): The message chain including system, human, and AI messages.
            - trace_id (str, optional): The root trace ID if resolved from the state.
            - iterations (int): Always returns 1, indicating the agent processes one iteration per call.
    -------------------------------------------------------
    """
    new_messages = []
    logger.info(f"Starting agent_node with trace_id: {state.get('trace_id')}")
    trace_id = None
    if state.get("trace_id") is None:
        run_tree = get_current_run_tree()
        trace_id = str(run_tree.trace_id) if run_tree else None
        logger.info(f"Resolved root trace_id: {trace_id}")

    task = (
        state.get("user_input")
        if state.get("router_decision") is None
        else state.get("router_decision").message
    )
    logger.debug(f"Task for agent: {task}")

    if state.get("iterations") < 1:
        logger.info(
            "First iteration detected. Returning initial tool call to navigate to google.com"
        )
        return {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                *state.get("user_mesg"),
                AIMessage(content="", tool_calls=[initial_tool_call]),
            ],
            "iterations": 1,
        }

    sc_message = await capture_and_build_screen_message(
        playwright_tools=playright_tools
    )

    try:
        rag_hits = rag_store.query(task, k=3) if rag_store else []
        rag_tips = [item["content"] for item in rag_hits]
        logger.info(f"Retrieved {len(rag_hits)} rag hits")

        if len(rag_hits) > 0:
            formatted = "\n".join(f"- {tip}" for tip in rag_tips)
            rag_tips = "Prior Guidance:\n" f"{formatted}\n"
            new_messages.append(HumanMessage(content=rag_tips))
    except Exception as e:
        logger.error(f"Failed to retrieve tips {e}")

    # Create the model
    # tools = [convert_to_openai_tool(tool) for tool in tools]
    llm = ChatQwen(
        api_base=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
        model=state.get("model", "browser-use-9b"),
        streaming=True,
        enable_thinking=True,
    )
    llm_with_tools = llm.bind_tools(tools=tools)
    logger.debug("Model initialized with tools and streaming")

    # # Invoke the Agent
    # if state.get("router_decision") is not None:
    #     decision = state.get("router_decision").message
    #     new_messages.append(AIMessage(content=decision))
    #     logger.debug(f"Previous Router Message added to message chain: {decision}")

    prompt_chain = state["messages"] + new_messages + [sc_message]
    logger.debug(f"Sending message chain of length: {len(prompt_chain)}")
    response = await llm_with_tools.ainvoke(prompt_chain)

    # Pull out the reasoning content and rebuild response message
    reasoning_text = response.additional_kwargs.get(
        "reasoning_content"
    ) or response.additional_kwargs.get("reasoning")
    content_blocks = []

    if response.tool_calls:
        for tool in response.tool_calls:
            content_blocks.append(
                {
                    "type": "tool_call",
                    "name": tool["name"],
                    "args": tool["args"],
                    "id": tool.get("id"),
                }
            )
    elif response.content:
        content_blocks.append({"type": "text", "text": response.content})
    if reasoning_text:
        content_blocks.append({"type": "reasoning", "reasoning": reasoning_text})

    response = AIMessage(
        content_blocks=content_blocks,
        tool_calls=response.tool_calls,
        invalid_tool_calls=response.invalid_tool_calls,
        usage_metadata=response.usage_metadata,
        id=response.id,
        response_metadata=response.response_metadata,
    )

    if trace_id:
        return {
            "messages": new_messages + [response],
            "trace_id": trace_id,
            "iterations": 1,
        }

    logger.info("agent_node finished successfully")
    return {"messages": new_messages + [response], "iterations": 1}


def router_node(state: AgentState):
    """
    -------------------------------------------------------
    Inspects the state to determine the next path
    -------------------------------------------------------
    Parameters:
       state - The current state of the graph (AgentState)
    Returns:
       router_output (RouterOutput)
    -------------------------------------------------------
    """
    logger.info("Entering router_node")
    # Check if the last message was a tool call
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            tool_name = get_tool_name_from_message(last_message, messages)
            tool_output = last_message.content
            logger.info(
                f"Last message is tool call: {tool_name} | output: {tool_output[:50]}"
            )
            if tool_name == "terminate":
                logger.info("Tool call 'terminate' detected, returning SUCCEEDED")
                return {
                    "router_decision": RouterOutput(
                        status="SUCCEEDED", message=tool_output
                    )
                }
            elif tool_name == "ask_user_question":
                logger.info(
                    "Tool call 'ask_user_question' detected, returning AWAITING_INPUT"
                )
                return {
                    "router_decision": RouterOutput(
                        status="AWAITING_INPUT", message=tool_output
                    )
                }
            else:
                logger.info("Other tool call detected, returning CONTINUE")
                return {
                    "router_decision": RouterOutput(
                        status="CONTINUE", message=tool_output
                    )
                }

    llm = ChatQwen(
        api_base=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
        model=state.get("model", "browser-use-9b"),
        streaming=True,
    )
    runnable = llm.with_structured_output(
        schema=RouterOutput, include_raw=False, method="json_mode"
    )
    prompt = ROUTER_PROMPT.format(user_input=state["user_input"])
    logger.debug(f"Router prompt generated: {prompt[:100]}...")
    messages = [
        *state.get("messages", []),
        HumanMessage(content=prompt),
    ]

    decision = runnable.invoke(messages)
    logger.info(
        f"Router decision: {decision.status} | message: {decision.message[:50]}"
    )
    return {"router_decision": decision}


def get_tool_name_from_message(
    tool_msg: ToolMessage, message_history: List[BaseMessage]
) -> str:
    """
    -------------------------------------------------------
    Finds the tool name by matching tool_call_id with prior AIMessages.
    -------------------------------------------------------
    Parameters:
       tool_msg - Tool message we need to find the Tool name of [ToolMessage]
       message_history - In order account of all prior messages (List)
    Returns:
       tool_name - the name of the tool correlated with tool_msg (str)
    -------------------------------------------------------
    """
    target_id = tool_msg.tool_call_id
    for msg in message_history[::-1]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["id"] == target_id:
                    return tool_call["name"]
    return None
