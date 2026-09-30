"""
-------------------------------------------------------
[Program Description]
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from .logger import get_logger
from pydantic import BaseModel

# Constants
load_dotenv()
logger = get_logger(__name__, "info")


def createOpenAIClient():
    """
    -------------------------------------------------------
    Initializes and returns an OpenAI client using API keys and
    base URLs from environment variables.
    -------------------------------------------------------
    Returns:
       client - The initialized OpenAI client object (OpenAI type)
    -------------------------------------------------------
    """

    return OpenAI(base_url=os.getenv("API_BASE_URL"), api_key=os.getenv("API_KEY"))


def runPrompt(
    client: OpenAI,
    messages: List,
    system_prompt: str,
    response_format: BaseModel = None,
    temperature: float = 1,
):
    """
    -------------------------------------------------------
    Executes a prompt using the provided client, parses the response,
    and logs the reasoning and final conclusion.
    -------------------------------------------------------
    Parameters:
       client - The initialized OpenAI client instance (OpenAI type)
       messages - The list of messages containing the prompt input (List type)
       system_prompt - ASIA (str)
       response_format - An optional response structure definition (BaseModel or None)
       temperature - The temperature setting for the LLM generation (float)
    Returns:
       output - The parsed output data from the LLM response (Type determined by response structure)
    -------------------------------------------------------
    """
    try:
        response = client.responses.parse(
            model=os.getenv("EXTRACTION_MODEL", "gemma4-opus"),
            instructions=system_prompt,
            input=messages,
            text_format=response_format,
            reasoning={"effort": "medium"},
            temperature=temperature,
        )
        reasoning = response.step_by_step_thinking
        logger.info(f"Reasoning output of llm call: {reasoning}")
        conlusion = response.final_conclusion
        logger.info(f"Conclusion of llm call: {conlusion}")
        output = response.output_parsed
    except Exception as e:
        logger.error(f"An error occurred while running prompt: {e}", exc_info=True)
        return None
    return output
