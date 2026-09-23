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
from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith.run_helpers import get_current_run_tree
from typing import List, Literal
from agent.prompt import SYSTEM_PROMPT, ROUTER_PROMPT
from .state import AgentState
from rag.store import TipStore
from utils.logger import get_logger

# Constants
logger = get_logger(__name__, "info")
rag_store: TipStore | None = TipStore()


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


async def agent_node(state: AgentState, tools: List):
    """
    -------------------------------------------------------
    Inspects the state to determine the next path.
    -------------------------------------------------------
    Parameters:
       [parameter name - parameter description (parameter type and constraints)]
    Returns:
       [return value name - return value description (return value type)]
    -------------------------------------------------------
    """
    logger.info(f"Starting agent_node with trace_id: {state.get('trace_id')}")
    trace_id = None
    if state.get("trace_id") is None:
        run_tree = get_current_run_tree()
        trace_id = run_tree.get_root().id if run_tree else None
        logger.info(f"Resolved root trace_id: {trace_id}")

    new_messages = []
    # Get the task and find any related rag tips
    task = (
        state.get("user_input")
        if state.get("router_decision") is None
        else state.get("router_decision").message
    )
    logger.debug(f"Task for agent: {task}")

    rag_hits = rag_store.query(task, k=3) if rag_store else []
    rag_tips = [item["content"] for item in rag_hits]
    logger.info(f"Retrieved {len(rag_hits)} rag hits")

    formatted = "\n".join(f"- {tip}" for tip in rag_tips)
    rag_tips = "Prior Guidance:\n" f"{formatted}\n"
    # Create the model
    tools = [convert_to_openai_tool(tool) for tool in tools]
    llm = ChatOpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
        model=state.model,
        streaming=True,
        stream_options={"include_usage": True},
    )
    llm_with_tools = llm.bind_tools(tools=tools)
    logger.debug("Model initialized with tools and streaming")

    # Invoke the Agent
    new_messages = (
        []
        if len(state.get("messages", [])) > 0
        else [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]
    )
    if state.get("router_decision") is not None:
        decision = state.get("router_decision").message
        new_messages.append(AIMessage(content=decision))
        logger.debug(f"Previous Router Message added to message chain: {decision}")
    new_messages.append(SystemMessage(content=rag_tips))

    prompt_chain = state["messages"] + new_messages
    logger.debug(f"Sending message chain of length: {len(prompt_chain)}")
    response = await llm_with_tools.ainvoke(prompt_chain)

    if trace_id:
        return {"messages": new_messages + [response], "trace_id": trace_id}
    
    logger.info("agent_node finished successfully")
    return {"messages": new_messages + [response]}


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
    llm = ChatOpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
        model=state.model,
        streaming=True,
        stream_options={"include_usage": True},
    )
    runnable = llm.with_structured_output(
        schema=RouterOutput, include_raw=False, method="json_mode"
    )
    prompt = ROUTER_PROMPT.format(
        user_input=state["user_input"], prev_message=state["messages"][-1]
    )
    logger.debug(f"Router prompt generated: {prompt[:100]}...")
    messages = [
        SystemMessage(content=prompt),
    ]
    
    decision = runnable.invoke(messages)
    logger.info(f"Router decision: {decision.status} | message: {decision.message[:50]}")
    return {"router_decision": decision}
