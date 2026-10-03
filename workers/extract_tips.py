"""
-------------------------------------------------------
Utilities for extracting and normalizing structured
tips from agent trajectory analyses.
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import os
from dotenv import load_dotenv
from utils.utils import createOpenAIClient, runPrompt
from .prompts import TipList, TIP_EXTRACTION_PROMPT
from utils.logger import get_logger

# Constants
project = os.getenv("LANGCHAIN_PROJECT")
load_dotenv()
logger = get_logger(__name__, "info")


def extract_structured_tips(domain, analysis_text):
    """
    -------------------------------------------------------
    Uses an LLM to extract structured tip information from
    the provided analysis text of an agent's trajectory.
    -------------------------------------------------------
    Parameters:
       domain (str): The specific domain categorization for the tips.
       analysis_text (str): The text containing the analysis of the agent trajectory.
    Returns:
       resp (List[Tips]): A dictionary representation of the extracted structured tips.
    -------------------------------------------------------
    """
    logger.info(f"Extracting Structured Tips for domain: {domain}")
    llmClient = createOpenAIClient()
    output: TipList = runPrompt(
        llmClient,
        [
            {
                "role": "user",
                "content": f"Target Domain: {domain}\n\nAgent trajectory Analysis:\n{str(analysis_text)}",
            }
        ],
        system_prompt=TIP_EXTRACTION_PROMPT,
        response_format=TipList,
        temperature=0.2,
    )
    logger.info(f"Tip Extraction Complete")
    return output.tips


def normalize_structured_tips(tip, domain, trajectory_id, outcome, description):
    """
    -------------------------------------------------------
    Normalizes a raw tip candidate object into a structured Tip format.
    -------------------------------------------------------
    Parameters:
       tip (Dict[str, Any]): The raw tip candidate data.
       domain (str): The domain categorization for the tip.
       trajectory_id (str): The ID of the trajectory the tip relates to.
       outcome (str): The final outcome observed.
       description (str): The description of the observation.
    Returns:
       Dict[str, Any]: The normalized structured tip dictionary.
    -------------------------------------------------------
    """
    steps = []
    if isinstance(tip.get("steps"), list):
        steps = [str(s) for s in tip["steps"] if s]

    tags = []
    if isinstance(tip.get("tags"), list):
        tags = [str(t).lower() for t in tip["tags"] if t]

    effectiveness = tip.get(
        "effectiveness", {"applied_count": 0, "success_count": 0, "last_applied": None}
    )
    # Return normalized structure
    return {
        "category": tip.get("category", "strategy"),
        "priority": tip.get("priority", "medium"),
        "domain": tip.get("domain") or domain,
        "content": tip.get("content") or "",
        "purpose": tip.get("purpose") or "",
        "trigger": tip.get("trigger") or "",
        "steps": steps,
        "negative_example": tip.get("negative_example") or "",
        "source": {
            "trajectory_id": trajectory_id,
            "outcome": outcome,
            "description": description or "",
        },
        "tags": tags,
        "effectiveness": effectiveness,
    }
