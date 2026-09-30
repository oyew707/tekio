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
from dotenv import load_dotenv
from utils.utils import createOpenAIClient, runPrompt
from prompts import Tip, TipList, TIP_EXTRACTION_PROMPT
from utils.logger import get_logger

# Constants
project = os.getenv("LANGCHAIN_PROJECT")
load_dotenv()
logger = get_logger(__name__, "info")


def extract_structured_tips(thoughts, domain, analysis_text):
   """
   -------------------------------------------------------
   [Function Description]
   -------------------------------------------------------
   Parameters:
      [parameter name - parameter description (parameter type and constraints)]
   Returns:
      [return value name - return value description (return value type)]
   -------------------------------------------------------
   """
   llmClient = createOpenAIClient()
   output: TipList = runPrompt(
      llmClient,
      {
         "role": "user",
         "content": f'Target Domain: {domain}\n\nAgent trajectory Analysis:\n{str(analysis_text)}'
      },
      system_prompt=TIP_EXTRACTION_PROMPT,
      response_format=TipList,
      temperature=0.2
   )
   resp = output.model_dump()
   return resp

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
   if isinstance(tip.get('steps'), list):
       steps = [str(s) for s in tip['steps'] if s]
       
   tags = []
   if isinstance(tip.get('tags'), list):
       tags = [str(t).lower() for t in tip['tags'] if t]
       
   effectiveness = tip.get('effectiveness', {
       "applied_count": 0,
       "success_count": 0,
       "last_applied": None
   })
   # Return normalized structure
   return {
       "category": tip.get('category', 'strategy'),
       "priority": tip.get('priority', 'medium'),
       "domain": tip.get('domain') or domain,
       "content": tip.get('content') or '',
       "purpose": tip.get('purpose') or '',
       "trigger": tip.get('trigger') or '',
       "steps": steps,
       "negative_example": tip.get('negative_example') or '',
       "source": {
           "trajectory_id": trajectory_id,
           "outcome": outcome,
           "description": description or ''
       },
       "tags": tags,
       "effectiveness": effectiveness
   }