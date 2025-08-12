# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

from typing import Dict, List, Any, Optional, Tuple
import json
import time
import os, math
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.text import split_into_chunks
from synthetic_data_kit.utils.rag_processor import RAGProccesor
from synthetic_data_kit.utils.llm_processing import (
    parse_summary,
    parse_qa_pairs,
    parse_ratings,
    convert_to_conversation_format,
)
from synthetic_data_kit.utils.config import (
    load_config,
    get_generation_config,
    get_curate_config,
    get_rag_config,
    get_prompt,
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QACurator:
    def __init__(self, client: LLMClient, config_path: Optional[Path] = None):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client

        # Load config
        self.config_path = config_path
        self.config = load_config(config_path)

        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)
        self.batch_size = self.curate_config.get("batch_size", 32)
        self.inference_batch = self.curate_config.get("inference_batch", 32)
        self.rating_temperature = self.curate_config.get("temperature", 0.1)
        self.threshold = self.curate_config.get("threshold", 7.0)
        self.rag_config = get_rag_config(self.config)
        self.enable_rag = self.rag_config.get("enable_rag", False)
        self.chunk_size_in_token = self.generation_config.get("chunk_size", 256)
        self.token_to_char_ratio = self.generation_config.get("token_to_char_ratio", 6)
        self.max_input_chars = max(
            1000, (self.generation_config.get("max_tokens", 2000) * self.token_to_char_ratio) - 2000
        )
        self.chunk_size_in_chars = self.generation_config.get(
            "chunk_size", 256
        ) * self.generation_config.get("token_to_char_ratio", 6)

    def _curate_process(
        self,
        messages: List[List[Dict[str, str]]],
        batches: List[List[Dict[str, str]]],
        batch_start: int,
    ) -> Tuple[List, List]:
        # Get ratings for the batch
        logger.info(f"Sending batch request with {len(messages)} items")

        max_tries = 3
        for i in range(max_tries):
            try:
                batch_responses = self.client.batch_completion(
                    messages, temperature=self.rating_temperature, batch_size=self.inference_batch
                )
                if len(batch_responses) > 0:
                    logger.debug(f"Curate_process Batch_completion completed successfully with response: {batch_responses}")
                    break
            except Exception as e:
                logger.error(f"batch_completion attempt failed for {i}/{max_tries}, with error: {e}")

        filtered_pairs = []
        unfiltered_pairs = []

        # Process each response
        for j, response in enumerate(batch_responses):
            original_batch_index = batch_start + j
            if original_batch_index < len(batches):
                original_batch = batches[original_batch_index]

                # Parse the ratings with original batch for fallback
                try:
                    rated_batch = parse_ratings(response, original_batch)
                    all_valid = all(
                        "question" in pair and "answer" in pair and "rating" in pair
                        for pair in rated_batch
                    )
                    if all_valid:
                        # Process the rated batch
                        for pair in rated_batch:
                            if "rating" in pair:
                                rating = pair["rating"]
                                if int(float(rating)) >= self.threshold:
                                    filtered_pairs.append(pair)
                                else:
                                    unfiltered_pairs.append(pair)
                    else:
                        logger.error(
                            f"Error processing batch {original_batch_index+1}: {str(rated_batch)}"
                        )
                except Exception as e:
                    logger.error(f"curation error: {e}")
                    pass

        return filtered_pairs, unfiltered_pairs

    def curate(
        self,
        messages: List[List[Dict[str, str]]],
        batches: List[List[Dict[str, str]]],
        batch_start: int,
    ) -> Tuple[List, List]:

        ## Perform curation
        filtered_pairs, unfiltered_pairs = self._curate_process(
            messages=messages, batches=batches, batch_start=batch_start
        )

        ## Perform enrichment
        ## @TODO Add Graph Enrichment

        return filtered_pairs, unfiltered_pairs

    def enrich(self, qa_pairs: List[Dict[str, str]] = None) -> str:
        """
        This is to enrich the message batch with RAG context
        """
        if qa_pairs is None or len(qa_pairs) == 0:
            raise Exception(f"Failed to process qa enrichment, empty messages pairs")

        ragClient = RAGProccesor(self.client, self.config_path)

        logger.info(
            f"Processing QA enrichment, stage 1 content enrichment, with {len(qa_pairs)} QA pairs"
        )

        try:
            response = ragClient.enrichQAPair(qa_pairs, max_chars=self.max_input_chars)
            if response is not None and len(response) > 0 :
                qa_pairs = response
        except Exception as e:
            logger.error(f"Exception occurred during QA enrichment: {e}")
            raise Exception(f"Exception occurred during QA enrichment: {e}")

        return qa_pairs
