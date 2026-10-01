"""
-------------------------------------------------------
[Program Description]
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import json
from typing import Any, List
from utils.logger import get_logger
import pandas as pd
from utils.utils import createOpenAIClient, runPrompt
from workers.extract_tips import normalize_structured_tips
from .prompts import Tip, MERGE_TIP_PROMPT
from sklearn.cluster import AgglomerativeClustering

# Constants
logger = get_logger(__name__, "info")
SIMILARITY_THRESHOLD = 0.85


def consolidate_tips(store):
    """
    -------------------------------------------------------
    Accepts new tips and adds them to exisitng tips store by fethcing all 
    tips, clusters them to find semantic duplicates, merges them, and 
    updates the database.
    -------------------------------------------------------
    Parameters:
       store - object to interact with tip database (Tipstore)
    -------------------------------------------------------
    """
    logger.info("Starting consolidation process")
        
    # 1. Get all tips with embeddings
    tips_data = store.all_tips(include_embedding=True)
    if not tips_data:
        logger.info("No tips found to consolidate.")
        return

    clusters = cluster_tips(tips_data)
    
    for group in clusters:
        if len(group) < 2:
            continue
            
        logger.info(f"Merging cluster of size {len(group)}")
        
        merged_tip_dict = merge_tips(group)
        new_content = merged_tip_dict["content"]
        new_metadata = {k: v for k, v in merged_tip_dict.items() if k != "content"}
        
        all_trace_ids = []
        for t in group:
            tid = t.get("trace_id")
            if tid:
                # Split by comma in case they were previously merged
                all_trace_ids.extend([x.strip() for x in tid.split(",")])
        unique_trace_ids = list(set(all_trace_ids))

        for old_tip in group:
            store.delete_tip(old_tip["id"])
            
        store.embed_and_upsert(
            content=new_content,
            metadata=new_metadata,
            trace_id=unique_trace_ids
        )
        logger.info("Consolidation of cluster complete.")

def cluster_tips(tips: list[dict[str, Any]]):
    """
    -------------------------------------------------------
    Uses AgglomerativeClustering to group similar tips.
    -------------------------------------------------------
    Parameters:
       tips - tips to be clusters (list[dict[str, Any]])
    Returns:
       list[list[dict]]: A list of clusters, where each cluster 
            is a list of tip dictionaries.
    -------------------------------------------------------
    """
    logger.info("Clustering Tips")
    df = pd.DataFrame(tips)
    X = df["embedding"].to_numpy()
    clustering = AgglomerativeClustering(
        metric="cosine", linkage="complete", distance_threshold=1 - SIMILARITY_THRESHOLD,
        compute_full_tree=True
    ).fit(X)
    
    df["cluster"] = clustering.labels_
    grouped = df.groupby("cluster")
    return grouped

    
    
def merge_tips(tips) -> Tip :
    """
    -------------------------------------------------------
    Uses an LLM to merge multiple tips into one.
    -------------------------------------------------------
    Parameters:
       domain (str): The specific domain categorization for the tips.
       analysis_text (str): The text containing the analysis of the agent trajectory.
    Returns:
       resp (Dict[str, Any]): A dictionary representation of the extracted structured tips.
    -------------------------------------------------------
    """
    llmClient = createOpenAIClient()
    output: Tip = runPrompt(
        llmClient,
        {
            "role": "user",
            "content": json.dumps(tips),
        },
        system_prompt=MERGE_TIP_PROMPT,
        response_format=Tip,
        temperature=0.1,
    )
    resp = output.model_dump()
    return resp
