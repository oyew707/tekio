"""
-------------------------------------------------------
Extracts and analyzes agent execution trajectories from LangSmith runs.
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import os
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from langsmith import Client
from utils.logger import get_logger
from laya import Router
from prompts import TRAJECTORY_ANALYSIS_PROMPT, TrajectoryAnalyzerOutput
from utils.utils import createOpenAIClient, runPrompt

# Constants
project = os.getenv("LANGCHAIN_PROJECT")
load_dotenv()
logger = get_logger(__name__, "info")
router = Router()
thought_classification_questions = {
    "thought_type": {
        "type": "choice",
        "instructions": "Classify the following reasoning thought into one of the specified categories.",
        "criteria": {
            "regular_thinking": "Planning, normal reasoning, goal-oriented steps.",
            "reflection": "Reviewing past steps, summarizing progress, or self-assessment.",
            "recovery_thought": "Addressing an error, recovering from a failure, or dealing with unexpected issues.",
        },
    }
}
CLASS_MAP = {
    "regular_thinking": "thought",
    "reflection": "reflection",
    "recovery_thought": "recovery",
}


def parse_thoughts(trajectory_id) -> Tuple[List[Dict, Any], List[Dict]]:
    """
    -------------------------------------------------------
    Parses a specific LangSmith run to extract structured reasoning 
    steps and the raw message trajectory.
    -------------------------------------------------------
    Parameters:
        trajectory_id (str): The unique identifier of the LangSmith run.
    Returns:
        steps (list[dict]): A list of structured step dictionaries containing 
            index, type, and content.
        trajectory (list[dict]): A chronological list of all entities/messages 
            encountered during execution.
    -------------------------------------------------------
    """
    run = Client().read_run(trajectory_id, load_child_runs=True)

    steps = []
    trajectory = []
    for idx, child in enumerate(run.child_runs, 1):
        if err := getattr(child, "error", None):
            steps.append({"index": idx, "type": "error", "content": str(err)})
            continue

        if child.name not in {"agent", "tools"}:
            continue

        step_type, content_parts = None, []
        messages = child.outputs.get("messages", []) if child.outputs else []
        # 2. Process messages
        for msg in messages:
            m_type = msg.get("type")
            m_content = msg.get("content")
            m_reasoning = msg.get("reasoning")

            trajectory.append({"entity": m_type, "message": m_content or m_reasoning})

            if m_type == "ai" and m_reasoning:
                try:
                    res = router.predict(m_reasoning, thought_classification_questions)
                    step_type = CLASS_MAP[res["answers"]["thought_type"]["choice"]]
                    break  # Stop processing further messages for this child run
                except Exception as e:
                    logger.warning(
                        f"Laya classification failed: {e}. Falling back to 'agent'."
                    )
                    step_type = "agent"
                    content_parts.append(m_reasoning)
            elif m_type in {"ai", "tool"} and m_content:
                content_parts.append(m_content)

        # 3. Add to steps only if a type was resolved
        if step_type:
            steps.append(
                {
                    "index": idx,
                    "type": step_type,
                    "content": "\n".join(content_parts).strip(),
                }
            )

    return steps, trajectory


def outcome(steps, trajectory, domain="general"):
    """
    -------------------------------------------------------
    Performs a high-level qualitative analysis of a trajectory using 
    an LLM to identify outcomes, decision chains, and failure modes.
    -------------------------------------------------------
    Parameters:
        steps (list[dict]): The structured steps parsed from the trajectory.
        trajectory (list[dict]): The raw message/entity trajectory.
        domain (str): The context or domain of the task (e.g., 'coding', 'math'). 
            Defaults to "general".
    Returns:
        dict: A dictionary containing the structured analysis (via 
            TrajectoryAnalyzerOutput) and the original raw steps.
    -------------------------------------------------------
    """
    llmClient = createOpenAIClient()
    output: TrajectoryAnalyzerOutput = runPrompt(
        llmClient,
        {
            "role": "user",
            "content": f"Domain: {domain}\n\nAgent execution trajectory:\n{str(trajectory)}",
        },
        system_prompt=TRAJECTORY_ANALYSIS_PROMPT,
        response_format=TrajectoryAnalyzerOutput,
        temperature=0.1,
    )
    resp = output.model_dump()
    resp["raw_steps"] = steps
    return resp


def format_analysis_extraction(analysis, trajectory):
    """
    -------------------------------------------------------
    Converts the structured analysis into a semantically rich format
    suitable for a prompt extraction task.
    -------------------------------------------------------
    Parameters:
       analysis (TrajectoryAnalyzerOutput)- Structured analysis of the trajectory.
       trajectory (str) - The condensed original trajectory text.
    Returns:
       formatted_string (str) - The formatted context string.
    -------------------------------------------------------
    """

    sections = []
    sections.append(f"## Trajectory Outcome: {analysis.outcome or 'unknown'}")

    if analysis.thought_classification:
        sections.append("\n## Agent Reasoning Classification")
        for t in analysis.thought_classification:
            sections.append(f"- Step {t.step} [{t.type}] ({t.quality}): {t.summary}")

    if analysis.decision_chain:
        sections.append("\n## Critical Decision Chain")
        for d in analysis.decision_chain:
            sections.append(
                f"- Step {d.step} [{d.causal_role}]: {d.decision} \u2192 {d.consequence}"
            )

    if analysis.failure_chains:
        sections.append("\n## Failure Analysis (Root Cause Chains)")
        for f in analysis.failure_chains:
            sections.append(
                f"- Symptom at step {f.symptom_step}, root cause at step {f.root_cause_step}: {f.root_cause}"
            )
            if f.recovery_step:
                sections.append(
                    f"  Recovery at step {f.recovery_step}: {f.recovery_method}"
                )

    if analysis.efficiency_issues:
        sections.append("\n## Efficiency Issues")
        for e in analysis.efficiency_issues:
            steps_list = e.steps if e.steps else []
            steps_str = ",".join(map(str, steps_list)) if steps_list else "N/A"
            sections.append(
                f"- Steps {steps_str}: {e.issue} \u2192 Better: {e.better_approach}"
            )

    if analysis.subtask_phases:
        sections.append("\n## Subtask Phases (for cross-task transfer)")
        for p in analysis.subtask_phases:
            sections.append(f"- {p.phase} ({p.outcome}): {p.transferable_pattern}")

    # Include condensed original text for specific details the analysis might reference
    condensed = trajectory
    if len(trajectory) > 4000:
        condensed = trajectory[:3000] + "\n[...truncated...]\n" + trajectory[-1000:]

    sections.append("\n## Original Trajectory (condensed)")
    sections.append(condensed)

    return "\n".join(sections)
