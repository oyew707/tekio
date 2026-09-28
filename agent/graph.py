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
from .tools import BrowserTools
from .state import AgentState
from .node import agent_node, router_node
from utils.logger import get_logger

# Constants
logger = get_logger(__name__, "info")
MAX_STEPS = 10


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
    iterations = state["iterations"]
    logger.debug(f"should_continue check - iterations: {iterations}")

    if iterations > MAX_STEPS:
        logger.warning(f"Stopping at max iterations ({MAX_STEPS})")
        return "end"

    router_decision = state.get("router_decision")
    if router_decision is None:
        logger.debug("No router decision yet, continuing")
        return "continue"

    status = router_decision.status
    logger.debug(f"Router decision status: {status}")

    if status == "CONTINUE":
        return "continue"

    logger.info("Router decided to end the loop")
    return "end"


async def build_graph(adapter):
    """
    -------------------------------------------------------
    Constructs and compiles the LangGraph workflow.
    -------------------------------------------------------
    Parameters:
        adapater - (MCPAdapter)
    Returns:
       app - The compiled LangGraph application (CompiledStateGraph)
    -------------------------------------------------------
    """
    logger.info("Fetching Playwright tools from MCP client")
    playwright_browser_tools = await adapter.list_tools()
    bb = BrowserTools(playwright_tools=playwright_browser_tools)
    cu_mapped_tools = bb.get_tools()
    logger.debug(f"Loaded {len(cu_mapped_tools)} tools")

    workflow = StateGraph(AgentState)
    logger.debug("Initializing StateGraph with AgentState")

    # Add the nodes
    agent_node_with_tools = partial(
        agent_node, tools=cu_mapped_tools, playright_tools=playwright_browser_tools
    )
    workflow.add_node("agent", agent_node_with_tools)
    workflow.add_node("tools", ToolNode(cu_mapped_tools, handle_tool_errors=True))
    workflow.add_node("router", router_node)
    logger.info(f"Added all nodes")

    # Set the entry point
    workflow.set_entry_point("agent")
    logger.info("Set entry point to 'agent'")

    # Add edges
    workflow.add_edge("tools", "router")

    # Conditional edge after agent
    workflow.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", "__end__": "router"}
    )
    workflow.add_conditional_edges(
        "router", should_continue, {"continue": "agent", "end": END}
    )
    logger.info("Graph edges and conditional edges configured")

    return workflow.compile()
