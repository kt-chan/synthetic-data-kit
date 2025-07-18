# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

from typing import Dict, List, Any, Optional, Tuple
import json
import time
import os
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.text import split_into_chunks
from synthetic_data_kit.utils.rag_processor import (
    reset_collection,
    wrte_chunks,
)
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
    get_prompt,
)


class QAGenerator:
    def __init__(self, client: LLMClient, config_path: Optional[Path] = None):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client

        # Load config
        self.config = load_config(config_path)

        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)

    def split_article_into_chunks(self, document_text: str) -> List[str]:
        """Split text into chunks with optional overlap"""
        # Get generation config
        chunk_size = self.generation_config.get("chunk_size", 4000)
        overlap = self.generation_config.get("overlap", 200)
        # Split text into chunks
        chunks = split_into_chunks(document_text, chunk_size=chunk_size, overlap=overlap)
        return chunks

    def generate_summary(
        self, document_text: str, fileName: str = None, enable_rag: bool = False
    ) -> str:
        """Generate a summary of the document"""
        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"
        batch_size = self.generation_config.get("batch_size", 32)
        max_seq_len = self.generation_config.get("max_seq_len", 4000) - 1000

        # Split text into chunks
        chunks = self.split_article_into_chunks(document_text)

        # Get summary generation prompt template
        summary_prompt_template = get_prompt(self.config, "summary")
        messages = []

        if len(chunks) > 1:
            # Prepare all message batches for each chunk section summary
            all_messages = []
            for i, chunk in enumerate(chunks):
                # Format the prompt with summary and text
                messages = [
                    {"role": "system", "content": summary_prompt_template},
                    {"role": "user", "content": chunk},
                ]
                all_messages.append(messages)

            print(
                f"Cut a doc size of {len(document_text)} into {len(chunks)} chunks to generate summary..."
            )

            if verbose:
                print(f"Messages: {all_messages}")

            summaries = self.batch_inference(all_messages, chunks, parse_summary)

            """
            Write chunks with metadata into vector database
            because batch_inference may return empty data, we have to map chunkid to summary
            """
            if enable_rag:
                fileNames = [fileName for i in range(len(chunks))]
                metas = [
                    {"filename": f, "id": s["id"], "summary": s["data"]}
                    for f, s in zip(fileNames, summaries)
                ]
                rag_chunks = []
                rag_metas = []
                rag_summary = []
                for item in metas:
                    rag_chunks.append(chunks[item["id"]])
                    rag_metas.append({"filename": item["filename"], "summary": item["summary"]})
                    rag_summary.append(item["summary"])
                reset_collection()
                wrte_chunks(rag_chunks, rag_metas)

            summaries = list(map(lambda x: x.get("data"), summaries))
            combined_summary = "\n".join(summaries)

            # Get summary generation prompt template for consolidation
            messages = [
                {"role": "system", "content": summary_prompt_template},
                {"role": "user", "content": combined_summary[:max_seq_len]},
            ]
        else:
            messages = [
                {"role": "system", "content": summary_prompt_template},
                {"role": "user", "content": document_text[:max_seq_len]},
            ]

        print(f"Summarizing chunks sector output of {len(str(messages))} ...")
        consolidated_summary = self.client.chat_completion(
            messages, temperature=0.1  # Use lower temperature for summaries
        )

        return consolidated_summary.strip()

    def generate_qa_pairs(
        self,
        document_text: str,
        summary: str,
        num_pairs: int = 25,
        fileName: str = None,
        enable_rag: bool = False,
    ) -> List[Dict[str, str]]:
        """Generate QA pairs from the document using batched processing"""
        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"
        batch_size = self.generation_config.get("batch_size", 32)

        # Split text into chunks
        chunks = self.split_article_into_chunks(document_text)
        pairs_per_chunk = max(1, round(num_pairs / len(chunks)))

        print(f"Generating QA pairs...")
        print(f"Document split into {len(chunks)} chunks")
        print(f"With {pairs_per_chunk} QA pairs in a chunk")

        # Get QA generation prompt template
        qa_prompt_template = get_prompt(self.config, "qa_generation")

        # Prepare all message batches
        all_messages = []
        for i, chunk in enumerate(chunks):
            # Format the prompt with summary and text
            qa_prompt = qa_prompt_template.format(
                num_pairs=pairs_per_chunk, summary=summary[:1000], text=chunk
            )

            messages = [{"role": "system", "content": qa_prompt}]
            all_messages.append(messages)

        print(
            f"Processing {len(chunks)} chunks to generate {pairs_per_chunk} QA pairs per chunk..."
        )
        result = self.batch_inference(all_messages, chunks, parse_qa_pairs)
        return result

    def batch_inference(
        self, all_messages: str, chunks: List[str], taskFunc
    ) -> List[Dict[str, str]]:
        """Inference using batched processing"""
        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"
        temperature = self.generation_config.get("temperature", 0.7)
        batch_size = self.generation_config.get("batch_size", 32)
        # Set up progress tracking based on verbose mode
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
            generate_task = progress_ctx.add_task(
                f"Generating batch inference output", total=len(chunks)
            )
            progress_ctx.start()
        else:
            progress_ctx = None
            generate_task = None

        all_inference_outputs = []
        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_messages = all_messages[batch_start:batch_end]
            current_batch_size = len(batch_messages)

            batch_num = batch_start // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            # Simple progress indicator for non-verbose mode
            if not verbose:
                print(f"Processing batch {batch_num}/{total_batches}...", end="\r")
            else:
                print(
                    f"Processing batch {batch_num}/{total_batches} with {current_batch_size} chunks ..."
                )

            try:
                # Process the batch
                batch_responses = self.client.batch_completion(
                    batch_messages, temperature=temperature, batch_size=batch_size
                )

                # Process each response in the batch
                for j, response in enumerate(batch_responses):
                    chunk_index = batch_start + j
                    chunk_pairs = taskFunc(chunk_index, response)
                    if isinstance(chunk_pairs, list):
                        all_inference_outputs.extend(chunk_pairs)
                    else:
                        all_inference_outputs.append(chunk_pairs)

                    if verbose:
                        print(f"Generated {len(chunk_pairs)} pairs from chunk {chunk_index+1}")
                        if len(chunk_pairs) == 0:
                            print(f"Empty resultset found {batch_messages}")

                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)

            except Exception as e:
                if verbose:
                    print(f"  Error processing batch {batch_num}: {str(e)}")

                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)

        # Stop progress bar if in verbose mode
        if progress_ctx:
            progress_ctx.stop()

        # Clear the progress line in non-verbose mode
        if not verbose:
            print(" " * 80, end="\r")
            print("Batch processing complete.")

        # Always print summary information, even in non-verbose mode
        print(f"Generated {len(all_inference_outputs)} chunks output in total")
        return all_inference_outputs

    def rate_qa_pairs(
        self, qa_pairs: List[Dict[str, str]], summary: str, threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rate and filter QA pairs by quality"""
        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"

        if not qa_pairs:
            return [], {"total": 0, "filtered": 0, "retention_rate": 0, "avg_score": 0}

        # Get threshold from args, then config, then default
        if threshold is None:
            threshold = self.curate_config.get("threshold", 7.0)

        if verbose:
            print(f"Evaluating {len(qa_pairs)} pairs...")

        # Get rating config
        batch_size = self.curate_config.get("batch_size", 8)
        temperature = self.curate_config.get("temperature", 0.1)

        # Get rating prompt template
        rating_prompt_template = get_prompt(self.config, "qa_rating")

        # Process in batches
        batches = [qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)]

        rated_pairs = []
        total_score = 0

        # Create progress bar
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        with Progress(*progress_columns) as progress:
            rating_task = progress.add_task(f"Rating QA pairs", total=len(batches))

            for i, batch in enumerate(batches):
                if verbose:
                    print(f"Rating batch {i+1}/{len(batches)}...")
                batch_json = json.dumps(batch, indent=2)

                # Format the rating prompt with pairs
                rating_prompt = rating_prompt_template.format(pairs=batch_json)

                messages = [{"role": "system", "content": rating_prompt}]

                try:
                    response = self.client.chat_completion(messages, temperature=temperature)

                    rated_batch = parse_ratings(response)

                    for pair in rated_batch:
                        if "rating" in pair:
                            total_score += pair["rating"]
                            if pair["rating"] >= threshold:
                                rated_pairs.append(pair)

                except Exception as e:
                    if verbose:
                        print(f"Error rating batch {i+1}: {str(e)}")

                time.sleep(0.5)  # Avoid rate limits
                progress.update(rating_task, advance=1)

        # Calculate metrics
        metrics = {
            "total": len(qa_pairs),
            "filtered": len(rated_pairs),
            "retention_rate": round(len(rated_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
            "avg_score": round(total_score / len(qa_pairs), 1) if qa_pairs else 0,
        }

        # Always print summary information, even in non-verbose mode
        print(f"Keeping {len(rated_pairs)} out of {len(qa_pairs)} pairs (threshold: {threshold})")
        print(f"Average score: {metrics['avg_score']}")
        return rated_pairs, metrics

    def process_document(
        self, document_text: str, num_pairs: int = 25, fileName: str = None, verbose: bool = False
    ) -> Dict[str, Any]:
        """Process a document to generate QA pairs without rating"""
        # Set the verbose environment variable
        if verbose:
            os.environ["SDK_VERBOSE"] = "true"
        else:
            os.environ["SDK_VERBOSE"] = "false"

        enable_rag = self.curate_config.get("enable_rag", False)

        # Generate summary
        summary = self.generate_summary(document_text, fileName=fileName, enable_rag=enable_rag)

        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(
            document_text,
            summary=summary,
            num_pairs=num_pairs,
            fileName=fileName,
            enable_rag=enable_rag,
        )

        # Prepare result - no rating at this stage
        result = {"summary": summary, "qa_pairs": qa_pairs}

        return result
