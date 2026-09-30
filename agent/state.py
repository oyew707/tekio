"""
-------------------------------------------------------
Defines the State for the Resume Agent
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
from typing import Dict, Annotated, Optional, List
from langgraph.graph import MessagesState
import operator
from pydantic import BaseModel, Field

# Constants


class BeliefUpdate(BaseModel):
    """
    -------------------------------------------------------
    Update to the representation of the agent's current understanding
    -------------------------------------------------------
    Parameters:
        sub_goal - The current specific task being worked on (str)
        action_history_summary - A brief summary of recent actions taken (str)
        environment_hypotheses - Current hypothesis about the state of the environment/page (str)
    -------------------------------------------------------
    """

    sub_goal: Optional[str] = Field(
        description="The current specific task being worked on"
    )
    action_history_summary: Optional[str] = Field(
        description="A brief summary of recent actions taken"
    )
    environment_hypotheses: Optional[str] = Field(
        description="Current hypothesis about the state of the environment/page"
    )


class BeliefState(BeliefUpdate):
    """
    -------------------------------------------------------
    Structured representation of the agent's current understanding
    -------------------------------------------------------
    Parameters:
        user_goal - The high-level objective provided by the user (str)
        sub_goal - The current specific task being worked on (str)
        extracted_facts - Key information gathered so far (List[str])
        action_history_summary - A brief summary of recent actions taken (str)
        environment_hypotheses - Current hypothesis about the state of the environment/page (str)
    -------------------------------------------------------
    """

    user_goal: str = Field(description="The high-level objective provided by the user")
    extracted_facts: Annotated[list, operator.add] = Field(
        description="Key information gathered so far"
    )


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
       belief - Belief state to include in the AI run (str)
    -------------------------------------------------------
    """

    user_input: str
    iterations: Annotated[int, operator.add]
    router_decision: Dict
    model: str
    trace_id: str
    user_mesg: list
    belief: BeliefState = None
