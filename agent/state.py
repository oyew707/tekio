"""
-------------------------------------------------------
Defines the State for the Resume Agent
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
from typing import Dict, Annotated
from langgraph.graph import MessagesState
import operator

# Constants


class AgentState(MessagesState):
    """
    -------------------------------------------------------
    State of the Agent throughout the graph execution
    -------------------------------------------------------
    Parameters:
       user_input - The users original input
       iterations - Count of draft/review cycles (int)
       model - The LLM (str)
       router_decision - has the decision to continue or not and accompanying message (see node.py)
       trace_id - Langsmith trace
       messages - Historical messages of the AI (List)
       user_mesg - Message between user and AI (List)
       facts - Facts to include in AI run (str)
    -------------------------------------------------------
    """

    user_input: str
    iterations: Annotated[int, operator.add]
    router_decision: Dict
    model: str
    trace_id: str
    user_mesg: list
    facts: Annotated[list, operator.add]
