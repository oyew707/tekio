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
from agent.prompt import format_belief_state
from agent.state import BeliefState
from dotenv import load_dotenv
from langsmith import Client
from utils.logger import get_logger
import laya
from .prompts import TRAJECTORY_ANALYSIS_PROMPT, TrajectoryAnalyzerOutput
from utils.utils import createOpenAIClient, runPrompt

# Constants
project = os.getenv("LANGCHAIN_PROJECT")
load_dotenv()
logger = get_logger(__name__, "debug")
router = laya.load(os.path.join(os.getcwd(), "models/laya"))
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


def parse_thoughts(trajectory_id) -> Tuple[List[Dict[str, Any]], List[Dict]]:
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
    logger.info(f"Parsing trajectory run: {trajectory_id}")
    run = Client().read_run(trajectory_id, load_child_runs=True)

    steps = []
    trajectory = []
    for idx, child in enumerate(run.child_runs, 1):
        if err := getattr(child, "error", None):
            logger.debug(f"Child run {idx} has error: {err}")
            steps.append({"index": idx, "type": "error", "content": str(err)})
            continue

        if child.name not in {"agent", "tools"}:
            logger.warning(f"Child is not an Agent or tool {child.name=}")
            continue

        step_type, content_parts = None, []
        messages = child.outputs.get("messages", []) if child.outputs else []
        sub_task, belief = None, None
        if child.outputs.get("belief") is not None:
            belief: BeliefState = BeliefState.model_validate(
                child.outputs.get("belief")
            )
            sub_task = belief.sub_goal
        logger.debug(
            f"Processing {child.name} [{child.outputs.keys()}] with {str(child.outputs)[:50]=} "
        )
        # 2. Process messages
        for msg in messages:
            m_type = msg.get("type")
            m_content = msg.get("content")

            if m_type not in ["ai", "tool"]:
                continue
            trajectory.append(
                {
                    "entity": m_type,
                    "message": m_content,
                    "Belief State": format_belief_state(belief) if belief else "",
                }
            )
            if m_type == "ai":
                try:
                    m_reasoning = next(
                        content_block.get("reasoning")
                        for content_block in m_content
                        if content_block.get("type") == "reasoning"
                    )
                    res = router.predict(m_reasoning, thought_classification_questions)
                    step_type = CLASS_MAP[res["answers"]["thought_type"]["choice"]]
                    logger.info(f"Step {idx}: Classified as {step_type}")
                    break  # Stop processing further messages for this child run
                except StopIteration:
                    logger.warning("Failed to find reasoning in the AI output")
                    step_type = "agent"
                    content_parts.append(m_content)
                except Exception as e:
                    logger.warning(
                        f"Laya classification failed: {e}. Falling back to 'agent'."
                    )
                    step_type = "agent"
                    content_parts.append(m_reasoning)
            elif m_type in {"tool"} and m_content:
                step_type = (
                    "action"
                    if msg.get("name") not in ("terminate", "ask_user_question")
                    else "output"
                )
                logger.debug(f"Full Message: {msg=}")
                content_parts.append(m_content)

        # 3. Add to steps only if a type was resolved

        if step_type:
            steps.append(
                {
                    "index": idx,
                    "type": step_type,
                    "content": "\n".join(content_parts).strip(),
                    "sub_task": sub_task,
                    "belief": (
                        format_belief_state(belief) if belief is not None else None
                    ),
                }
            )
    logger.info(f"Parsed {len(steps)} steps and {len(trajectory)} trajectory entries")

    return steps, trajectory


def outcome(steps, domain="general"):
    """
    -------------------------------------------------------
    Performs a high-level qualitative analysis of a trajectory using
    an LLM to identify outcomes, decision chains, and failure modes.
    -------------------------------------------------------
    Parameters:
        steps (list[dict]): The structured steps parsed from the trajectory.
        domain (str): The context or domain of the task (e.g., 'coding', 'math').
            Defaults to "general".
    Returns:
        dict: A dictionary containing the structured analysis (via
            TrajectoryAnalyzerOutput) and the original raw steps.
    -------------------------------------------------------
    """
    logger.info(f"Performing outcome analysis for domain: {domain}")
    llmClient = createOpenAIClient()
    output: TrajectoryAnalyzerOutput = runPrompt(
        llmClient,
        [
            {
                "role": "user",
                "content": f"Domain: {domain}\n\nAgent execution trajectory:\n{str(steps)}",
            }
        ],
        system_prompt=TRAJECTORY_ANALYSIS_PROMPT,
        response_format=TrajectoryAnalyzerOutput,
        temperature=0.1,
    )
    resp = output.model_dump()
    resp["raw_steps"] = steps
    logger.info("Outcome analysis complete")
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
    logger.info(f"Formatting analysis extraction")

    sections = []
    sections.append(f"## Trajectory Outcome: {analysis.get("outcome", 'unknown')}")

    if analysis.get("thought_classification"):
        logger.info("Including thought classification")
        sections.append("\n## Agent Reasoning Classification")
        for t in analysis.get("thought_classification"):
            if not isinstance(t, dict):
                logger.warning(f"Unexpected thought item is not a dict {str(t)}")
                continue
            sections.append(
                f'- Step {t.get("step")} [{t.get("type")}] ({t.get("quality")}): {t.get("summary")}'
            )

    if analysis.get("decision_chain"):
        logger.info("Including decision chain")
        sections.append("\n## Critical Decision Chain")
        for d in analysis.get("decision_chain"):
            if not isinstance(d, dict):
                logger.warning(f"Unexpected Decision Chain item is not a dict {str(d)}")
                continue
            sections.append(
                f'- Step {d.get("step")} [{d.get("causal_role")}]: {d.get("decision")} \u2192 {d.get("consequence")}'
            )

    if analysis.get("failure_chains"):
        logger.info("Including failure chains")
        sections.append("\n## Failure Analysis (Root Cause Chains)")
        for f in analysis.get("failure_chains"):
            if not isinstance(f, dict):
                logger.warning(f"Unexpected Failure Chain item is not a dict {str(f)}")
                continue
            sections.append(
                f"- Symptom at step {f.get('symptom_step')}, root cause at step {f.get('root_cause_step')}: {f.get('root_cause')}"
            )
            if f.get("recovery_step"):
                sections.append(
                    f"\tRecovery at step {f.get('recovery_step')}: {f.get('recovery_method')}"
                )

    if analysis.get("efficiency_issues"):
        logger.info("Including efficiency issues")
        sections.append("\n## Efficiency Issues")
        for e in analysis.get("efficiency_issues"):
            if not isinstance(e, dict):
                logger.warning(
                    f"Unexpected Efficiency Issue item is not a dict {str(e)}"
                )
                continue
            steps_list = e.get("steps", [])
            steps_str = ",".join(map(str, steps_list)) if steps_list else "N/A"
            sections.append(
                f"- Steps {steps_str}: {e.get('issue')} \u2192 Better: {e.get('better_approach')}"
            )

    if analysis.get("subtask_phases"):
        logger.info("Including subtask phases")
        sections.append("\n## Subtask Phases (for cross-task transfer)")
        for p in analysis.get("subtask_phases"):
            if not isinstance(p, dict):
                logger.warning(f"Unexpected Subtask Phase item is not a dict {str(p)}")
                continue
            sections.append(
                f"- {p.get('phase')} ({p.get('outcome')}): {p.get('transferable_pattern')}"
            )

    # Include condensed original text for specific details the analysis might reference
    if not isinstance(trajectory, str):
        logger.warning(
            "trajectory is not a string, converting to string representation"
        )
        trajectory = str(trajectory)

    condensed = trajectory
    if len(trajectory) > 4000:
        condensed = trajectory[:3000] + "\n[...truncated...]\n" + trajectory[-1000:]

    sections.append("\n## Original Trajectory (condensed)")
    sections.append(condensed)

    return "\n".join(sections)
