"""
-------------------------------------------------------
LangGraph Orchestrator Definition
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from .state import AgentState
from .node import agent_node, router_node

# Constants
MAX_STEPS = 20
client = MultiServerMCPClient(
    {
        "playwright": {
            "url": "http://playwright-mcp:8931/sse",
            "transport": "sse",
            "headers": {"Host": "localhost:8931"},
        }
    }
)
AVAILABLE_AGENTIC_MODELS = [
    "browser-use-9b",
    "gemma4-12b-agentic",
]


def should_continue(state: AgentState) -> str:
    """
    -------------------------------------------------------
    Determines whether the graph should loop back to the agent
    or finish based on the iteration count or feedback.
    -------------------------------------------------------
    Parameters:
       state - The current state of the graph (AgentState)
    Returns:
       next_node - The string name of the next node or END (str)
    -------------------------------------------------------
    """
    # Stop after 3 iterations to prevent infinite loops
    if state["iterations"] > MAX_STEPS:
        return "end"

    if (
        state.get("router_decision") is None
        or state["router_decision"].status == "CONTINUE"
    ):
        return "continue"

    return "end"


async def build_graph():
    """
    -------------------------------------------------------
    Constructs and compiles the LangGraph workflow.
    -------------------------------------------------------
    Returns:
       app - The compiled LangGraph application (CompiledStateGraph)
    -------------------------------------------------------
    """
    playwright_browser_tools = await client.get_tools()

    workflow = StateGraph(AgentState)

    # Add the nodes
    agent_node_with_tools = partial(agent_node, tools=playwright_browser_tools)
    workflow.add_node("agent", agent_node_with_tools)
    workflow.add_node("tools", ToolNode(playwright_browser_tools))
    workflow.add_node("router", router_node)

    # Set the entry point
    workflow.set_entry_point("agent")

    # Add edges
    workflow.add_edge("agent", "router")
    workflow.add_edge("tools", "agent")

    # Conditional edge after agent
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_conditional_edges(
        "router", should_continue, {"continue": "agent", "end": END}
    )

    return workflow.compile()
