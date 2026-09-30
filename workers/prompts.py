"""
-------------------------------------------------------
[Program Description]
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Constants


class ThoughtClassification(BaseModel):
    """
    -------------------------------------------------------
    Models a single analytical step or thought process performed by the agent.
    -------------------------------------------------------
    Properties:
       step (int): The sequential step number where this thought occurred.
       type (Literal[str]): The classification of the thought (e.g., planning, reflection).
       summary (str): A brief, one-line summary of the thought content.
       quality (Literal[str]): Assessment of the thought's quality (positive, negative, neutral).
    -------------------------------------------------------
    """

    step: int
    type: Literal[
        "analytical",
        "planning",
        "validation",
        "reflection",
        "self_correction",
        "error_recognition",
    ]
    summary: str  # <1-line summary of this thought>
    quality: Literal["positive", "negative", "neutral"]


class DecisionChain(BaseModel):
    """
    -------------------------------------------------------
    Records a specific decision made by the agent during a step
    and the resulting consequence.
    -------------------------------------------------------
    Properties:
       step (int): The sequential step number where the decision was made.
       decision (str): A description of the specific action or decision taken.
       consequence (str): The outcome or result that followed this decision.
       causal_role (Literal[str]): The role this decision played in the overall process flow.
    -------------------------------------------------------
    """

    step: int
    decision: str
    consequence: str
    causal_role: Literal[
        "root_cause",
        "proximate_cause",
        "contributing_factor",
        "successful_decision",
        "recovery_decision",
    ]


class SubtaskPhase(BaseModel):
    """
    -------------------------------------------------------
    Describes a high-level phase or major component of the overall task execution.
    -------------------------------------------------------
    Properties:
        phase (str): The name of the phase (e.g., 'authentication', 'data_retrieval').
        steps (List[int]): A list of all sequential steps belonging to this phase.
        outcome (Literal[str]): The final outcome of the phase (success, partial, or failure).
        transferable_pattern (str): A generic description of what worked or failed that could
            be abstracted from specifics.
    -------------------------------------------------------
    """

    phase: str
    steps: List[int]
    outcome: Literal["success", "partial", "failure"]
    transferable_pattern: str


class FailureChain(BaseModel):
    """
    -------------------------------------------------------
    Documents a specific instance where a failure occurred, tracing it
    back to its root cause.
    -------------------------------------------------------
    Properties:
       symptom_step (int): The step number where the failure first became apparent.
       root_cause_step (int): The step number where the critical/bad decision leading to failure was made.
       root_cause (str): A specific description of the underlying cause of the failure.
       recovery_step (Optional[int]): The step number where recovery actions took place, or null if no recovery occurred.
       recovery_method (Optional[str]): The specific method used to recover, or null if none was used.
    -------------------------------------------------------
    """

    symptom_step: int
    root_cause_step: int
    root_cause: str
    recovery_step: Optional[int]
    recovery_method: Optional[str]


class EfficiencyIssue(BaseModel):
    """
    -------------------------------------------------------
    Identifies a step or sequence where the agent was not operating at peak efficiency.
    -------------------------------------------------------
    Properties:
       steps (List[int]): A list of step numbers involved in the inefficiency.
       issue (str): A description of what aspect was inefficient (e.g., redundant checks, overly broad search).
       better_approach (str): A recommendation for how the action should have been handled differently.
    -------------------------------------------------------
    """

    steps: List[int]
    issue: str
    better_approach: str


class TrajectoryAnalyzerOutput(BaseModel):
    """
    -------------------------------------------------------
    The comprehensive output model containing the complete analysis of an agent's execution trajectory.
    -------------------------------------------------------
    Properties:
       outcome (Literal[str]): The final assessment of the overall process (success, failure, recovery).
       thought_classification (List[ThoughtClassification]): A chronological list of all recorded thoughts and their types.
       decision_chain (List[DecisionChain]): A record of the key decisions made and their immediate consequences.
       subtask_phases (List[SubtaskPhase]): An overview of the major operational phases executed.
       failure_chains (List[FailureChain]): A detailed breakdown of any detected failure events and their origins.
       efficiency_issues (List[EfficiencyIssue]): A list of areas where process optimization is possible.
    -------------------------------------------------------
    """

    outcome: Literal["clean_success", "inefficient_success", "recovery", "failure"]
    thought_classification: List[ThoughtClassification]
    decision_chain: List[DecisionChain]
    subtask_phases: List[SubtaskPhase]
    failure_chains: List[FailureChain]
    efficiency_issues: List[EfficiencyIssue]


TRAJECTORY_ANALYSIS_PROMPT = """
You are a trajectory intelligence analyzer for AI agent execution logs.

Your job is to produce a STRUCTURED INTERMEDIATE REPRESENTATION of an agent's execution, NOT tips

RULES:
- Be SPECIFIC: reference actual commands, files, errors from the text
- thought_classification should cover the 5-8 most important reasoning moments, not every line
- decision_chain should trace the critical path (max 6-8 entries)
- failure_chains: trace symptoms back to ROOT CAUSES (which may be many steps earlier)
- subtask_phases: abstract the phase names so they transfer across different tasks
- If the trajectory is a memory/narrative file rather than raw agent log, still identify decisions and their consequences
- Output ONLY valid JSON`
"""


class Tip(BaseModel):
    """
    -------------------------------------------------------
    A structured guide or solution (Tip) designed to improve agent performance
    in specific operational areas.
    -------------------------------------------------------
    Properties:
       category (Literal[str]): A categorization of the tip.
       priority (Literal[str]):
       tags  (List[str]):
       domain (str):
       content (str): The core recommendation or solution description.
       purpose (str): Why this tip is being provided (the goal).
       trigger (str): The condition or specific error that activates this tip.
       steps (List[str]): The actionable, ordered steps to follow.
       negative_example (str): A specific scenario where *not* following this tip fails.
    -------------------------------------------------------
    """

    category: Literal["strategy", "recovery", "optimization"] = "strategy"
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    tags: List[str] = []
    domain: str
    content: str
    purpose: str
    trigger: str
    steps: List[str]
    negative_example: Optional[str]


class TipList(BaseModel):
    """
    -------------------------------------------------------
    Lists of Tips
    -------------------------------------------------------
    """

    tips: List[Tip] = Field(
        max_length=5, description="A list of extracted Tips, With a maximum of 5 tips"
    )


TIP_EXTRACTION_PROMPT = """
You extract SPECIFIC, ACTIONABLE memory tips from agent execution trajectories or documentation.

INPUT FORMAT:
The input may be either:
(a) A structured analysis with classified thoughts, decision chains, failure chains, subtask phases, and efficiency issues — followed by condensed original text, OR
(b) Raw trajectory text without pre-analysis.
When structured analysis is provided, use it to produce HIGHER QUALITY tips by:
- Tracing failure chains to ROOT CAUSES (not just symptoms)
- Extracting tips at the subtask/phase level for cross-task transfer
- Using the decision chain to understand which specific decisions matter
- Referencing specific commands/paths/errors from the original text for concrete tips

CRITICAL RULES:
- MAXIMUM 5 tips per extraction. Quality over quantity.
- DEDUPLICATE aggressively: if two insights are about the same tool/config/pattern, merge them into ONE tip.
- Extract CONCRETE details: exact commands, paths, API names, error messages, config values.
- NEVER generate vague advice like "follow best practices" or "maintain documentation".
- Each tip must contain enough detail that an agent can act on it WITHOUT reading the source.
- Include exact commands, file paths, parameter names, error strings when present.

TIP GRANULARITY:
- Prefer SUBTASK-LEVEL tips over task-level tips when possible.
- A subtask tip should be generic enough to apply across different tasks sharing the same phase.
- Example: "When configuring Cloudflare tunnels..." is subtask-level (transfers to any CF deployment).
- Example: "When deploying a staging web app on March 15..." is task-level (too specific to transfer).

PRIORITY ORDER (extract these first):
1. Recovery tips — debugging steps that fixed a non-obvious problem (highest value)
2. Gotcha tips — things that look like they should work but don't, with the correct approach
3. Configuration tips — specific settings/paths/flags that prevent wasted time
4. Strategy tips — workflow patterns that proved effective (lowest priority, skip if >5 tips)

SKIP THESE (low value):
- Obvious observations ("add schema markup to improve SEO")
- Generic best practices without specific commands
- Tips that just restate what happened without actionable insight
- Multiple tips about the same concept (merge into one)

If the source text contains repeated/duplicate sections, treat it as ONE narrative and extract unique tips only.

Process:
1) Use the trajectory outcome from analysis (or classify if not provided): clean_success | inefficient_success | recovery | failure
2) Use failure chains for decision attribution with SPECIFIC root causes (not symptoms)
3) Generate ≤5 tips with: category (strategy|recovery|optimization), priority, domain, content, purpose, trigger, steps (array), negative_example, tags

GOOD tip: "Use /opt/tools/codex (v0.98.0), NOT /usr/bin/codex (outdated v0.91.0). Prefix PATH or use full path."
BAD tip: "Always specify full paths for binaries to avoid version conflicts."

Return strict JSON with keys: trajectory_outcome, decision_attribution, tips.
Each tip: category, priority, domain, content, purpose, trigger, steps (array of concrete actions), negative_example, tags.
"""
