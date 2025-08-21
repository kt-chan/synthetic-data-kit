# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Filter low quality examples

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_curator import QACurator
from synthetic_data_kit.utils.config import get_curate_config, get_prompt, get_rag_config
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format, parse_ratings
from synthetic_data_kit.utils.app_logger import get_logger

logger = get_logger(__name__)

def curate_qa_pairs(
    input_path: str,
    output_path: str,
    threshold: Optional[float] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[Path] = None,
    provider: Optional[str] = None,
    verbose: Optional[bool] = False
) -> str:
    """Clean and filter QA pairs based on quality ratings

    Args:
        input_path: Path to the input file with QA pairs
        output_path: Path to save the cleaned output
        threshold: Quality threshold (1-10)
        api_base: VLLM API base URL
        model: Model to use
        config_path: Path to configuration file
        verbose: Show detailed output

    Returns:
        Path to the cleaned output file
    """
    # Set verbose either via CLI or via env variable. If its via CLI, set it to env variable
    if verbose:
        os.environ["SDK_VERBOSE"] = "true"
    else:
        os.environ["SDK_VERBOSE"] = "false"

    # Load input file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract QA pairs
    qa_pairs = data.get("qa_pairs", [])
    summary = data.get("summary", "")

    # If there are no QA pairs or they're already filtered
    if not qa_pairs or len(qa_pairs) == 0:
        raise ValueError("No QA pairs found in the input file")

    # Initialize LLM client
    client = LLMClient(
        config_path=config_path, provider=provider, api_base=api_base, model_name=model
    )

    # Initialize QA Curator
    qaCuratorClient = QACurator(client, config_path)

    # Get configuration
    curate_config = get_curate_config(client.config)
    rag_config = get_rag_config(client.config)
    enable_rag = rag_config.get("enable_rag", False)

    batch_size = curate_config.get("batch_size", 32)
    inference_batch = curate_config.get("inference_batch", 32)
    threshold = curate_config.get("threshold", 7.0) if threshold is None else threshold

    # Get rating prompt template
    rating_prompt_template = get_prompt(client.config, "qa_rating")

    # Split QA pairs into batches
    batches = []
    for i in range(0, len(qa_pairs), batch_size):
        try:
            batch = qa_pairs[i : i + batch_size]
            if isinstance(batch, list) and len(batch) > 1:
                if batch[0]["question"] is not None and batch[0]["answer"] is not None:
                    batches.extend(batch)
            if isinstance(batch, dict):
                if batch["question"] is not None and batch["answer"] is not None:
                    batches.append([batch])
        except Exception as e:
            logger.error(f"Parsing Error for {e}")
            pass

    # Prepare all message batches for rating
    all_messages = []

    # Enrich with RAG Content
    if enable_rag:
        response = qaCuratorClient.enrich(batches)
        if response is not None and len(response) > 0:
            batches = response
        else:
            logger.error(f"Error in enriching QA Content for {batches}")

    for batch in batches:
        rating_prompt = rating_prompt_template.format(
            question=batch["question"],
            answer=batch["answer"],
            content=batch.get("content", "There is no reference context exist."),
        ).strip()
        messages = [{"role": "system", "content": rating_prompt}]
        all_messages.append(messages)

    # Initialize counters and result containers
    total_filtered_pairs = []
    total_unfiltered_pairs = []

    # Process batches with simple progress indicator rather than a detailed bar
    # This avoids conflicts with other output messages
    print(f"Processing {len(batches)} batches of QA pairs...")

    # Only use detailed progress bar in verbose mode
    if verbose:
        from rich.progress import (
            Progress,
            BarColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        progress_ctx = Progress(*progress_columns)
        rate_task = progress_ctx.add_task(f"Rating QA pairs", total=len(batches))
        progress_ctx.start()
    else:
        progress_ctx = None
        rate_task = None

    logger.info(f"Curating total {len(batches)} qa-pairs...")

    # Process in inference batches
    for batch_start in range(0, len(all_messages), inference_batch):
        batch_end = min(batch_start + inference_batch, len(all_messages))
        current_batch = all_messages[batch_start:batch_end]
        current_batch_size = len(current_batch)

        batch_num = batch_start // inference_batch + 1
        total_batches = (len(all_messages) + inference_batch - 1) // inference_batch
        # Simple progress indicator for non-verbose mode
        logger.info(
            f"Curating batch {batch_num}/{total_batches} with {current_batch_size} chunks each ..."
        )

        try:

            filtered_pairs, unfiltered_pairs = qaCuratorClient.curate(
                messages=current_batch, batches=batches, batch_start=batch_start
            )

            if filtered_pairs is not None and len(filtered_pairs) > 0:
                total_filtered_pairs.extend(filtered_pairs)

            if unfiltered_pairs is not None and len(unfiltered_pairs) > 0:
                total_unfiltered_pairs.extend(unfiltered_pairs)

            # Update progress bar if in verbose mode
            if progress_ctx and rate_task:
                progress_ctx.update(rate_task, advance=current_batch_size)

        except Exception as e:
            if verbose:
                logger.error(f"Error processing inference batch {batch_num}: {str(e)}")

            # Update progress bar if in verbose mode
            if progress_ctx and rate_task:
                progress_ctx.update(rate_task, advance=current_batch_size)

    # Stop progress bar if in verbose mode
    if progress_ctx:
        progress_ctx.stop()

    # Clear the progress line in non-verbose mode
    if not verbose:
        logger.info("Batch processing complete.")

    # Calculate Scores
    total_score = 0
    total_evaluated = len(total_filtered_pairs) + len(total_unfiltered_pairs)
    total_passed = len(total_filtered_pairs)

    for idx, score in enumerate(total_filtered_pairs):
        if score["rating"] is not None and isinstance(score["rating"], int):
            total_score += score["rating"]

    # Calculate metrics
    metrics = {
        "total": len(batches),
        "filtered": len(total_filtered_pairs),
        "evalutated": total_evaluated,
        "retention_rate": round(len(total_filtered_pairs) / len(batches), 2) if batches else 0,
        "avg_score": round(total_score / total_evaluated, 1) if total_evaluated else 0,
    }

    # Always print basic stats, even in non-verbose mode
    logger.info(f"Rated {total_evaluated} QA pairs")
    logger.info(f"Retained {total_passed} pairs (threshold: {threshold})")
    logger.info(f"Average score: {metrics['avg_score']}")

    # Convert to conversation format
    conversations = convert_to_conversation_format(total_filtered_pairs)

    # Create result with filtered pairs
    result = {
        "summary": summary,
        "qa_pairs": total_filtered_pairs,
        "conversations": conversations,
        "bad_qa_pairs": total_unfiltered_pairs,
        "metrics": metrics,
    }

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return output_path
