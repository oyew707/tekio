"""
-------------------------------------------------------
Graph Nodes (Actor and Reviewer)
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
from pydantic import BaseModel, Field
from typing import Literal, Optional, Annotated
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith.run_trees import get_current_run_tree
from typing import Any, List, Literal
from agent.prompt import SYSTEM_PROMPT, ROUTER_PROMPT
from .state import AgentState
from ..rag.store import TipStore

# COnstants
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
    message: str = Field(description="message to Agent to continue its workflow or to user if complete")
    status: Literal['CONTINUE', 'SUCCEEDED', 'FAILED', 'AWAITING_INPUT']
    

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
    trace_id = None
    if state.get("trace_id") is None:
        run_tree = get_current_run_tree()
        trace_id = run_tree.get_root().id if run_tree else None
    
    new_messages = []
    # Get the task and find any related rag tips
    task = state.get("user_input") if state.get("router_decision") is None else state.get("router_decision").message
    
    rag_hits = rag_store.query(task, k=3) if rag_store else []
    rag_tips = [item["content"] for item in rag_hits]
    
    formatted = "\n".join(f"- {tip}" for tip in rag_tips)
    rag_tips =  (
        "Prior Guidance:\n"
        f"{formatted}\n"
    )
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
    
    # Invoke the Agent
    new_messages = [] if len(state.get("messages", [])) > 0 else [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=task)
    ]
    if state.get("router_decision") is not None:
        new_messages.append(AIMessage(content = state.get("router_decision").message))
    new_messages.append(SystemMessage(content = rag_tips))
    
    response = await llm_with_tools.ainvoke(state["messages"] + new_messages)
    
    if trace_id:
        return {"messages": new_messages + [response], "trace_id": trace_id}
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

    llm = ChatOpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
            model=state.model,
            streaming=True,
            stream_options={"include_usage": True},
        )
    runnable = llm.with_structured_output(schema=RouterOutput, include_raw=False, method='json_mode')
    prompt = ROUTER_PROMPT.format(user_input = state["user_input"], prev_message=state["messages"][-1])
    messages = [
        SystemMessage(content=prompt),
    ]
    return {"router_decision": runnable.invoke(messages)}

    