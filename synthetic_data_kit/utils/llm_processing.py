# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Output utilities
import re
import json
import os
from typing import List, Dict, Any, Optional
from synthetic_data_kit.utils.app_logger import get_logger

logger = get_logger(__name__)


def parse_summary(chunk_index: int, chunk: str) -> List[Dict[str, str]]:
    """Parse chunk summary from LLM output with enhanced error handling"""
    verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"
    logger.debug(f"Parsing response of length {len(chunk)}")

    if chunk and len(chunk) > 0:
        cleaned_text = re.sub(
            r"(\n\s*|\r\s*)", " ", chunk
        ).strip()  # Remove newlines and extra spaces
        cleaned_text = re.sub(
            r",(\s*\}|\s*\])", r"\1", cleaned_text
        ).strip()  # Remove trailing commas
    return [{"id": chunk_index, "data": cleaned_text}]


def parse_qa_pairs(chunk_index: int, text: str) -> List[Dict[str, str]]:
    """Parse QA pairs from LLM output with enhanced error handling"""
    verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"

    if verbose:
        logger.info(f"Parsing response of length {len(text)}")

    try:
        # Try direct JSON parsing
        if "[" in text:
            # Find the first [ and last ]
            start = text.find("[")
            end = text.rfind("]")

            # If there is no closing ], append it
            if end == -1:
                end = text.rfind("},")
                json_text = text[start : end + 1] + "]"
            else:
                json_text = text[start : end + 1]

            # Try to clean up the JSON to fix common issues
            cleaned_text = re.sub(
                r"(\n\s*|\r\s*)", " ", json_text
            )  # Remove newlines and extra spaces
            cleaned_text = re.sub(r",(\s*\}|\s*\])", r"\1", cleaned_text)  # Remove trailing commas

            try:
                pairs = json.loads(cleaned_text)
                return pairs
            except json.JSONDecodeError as e:
                logger.info(f"Direct JSON parsing failed: {e}")
                ## Fix Json Format for Missing Double Quote in Text String Closing.
                text = cleaned_text
                start = text.find("[")
                end = text.rfind("]")
                list_start = text.find("{")
                list_end = text.rfind("}")
                if text[list_end] != '"':
                    cleaned_text = text[:list_end] + '"' + text[list_end:]
                pairs = json.loads(cleaned_text)
                return pairs

        # If all fixes fail, return original with warning
        raise Exception(f"Error during JSON extraction")
    except Exception as e:
        logger.error(f"Falling back to regex pattern matching, Error during JSON extraction: {e}")

    qa_pattern = r'"question":\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer":\s*"((?:[^"\\]|\\.)*)"\s*'
    pairs = []

    for match in re.finditer(qa_pattern, text):
        try:
            q = match.group(1).replace('\\"', '"')
            a = match.group(2).replace('\\"', '"')
            pairs.append({"question": q, "answer": a})
        except Exception as e:
            if verbose:
                logger.info(f"Error extracting pair: {e}")

    if verbose:
        if pairs:
            logger.info(f"Extracted {len(pairs)} QA pairs with regex")
        else:
            logger.info("No QA pairs extracted. Check the model output format.")

    return pairs


def parse_ratings(text: str, original_items: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Parse rated items from LLM output

    Attempts to parse JSON from LLM response. Will raise an exception if
    parsing fails. Never adds default ratings - either the model returns valid
    ratings or the function will crash.

    Args:
        text: LLM response text to parse
        original_items: Original QA pairs (ignored - no defaults used)

    Returns:
        List of items with ratings from the LLM

    Raises:
        ValueError: If the response cannot be parsed as valid JSON
    """
    verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"

    if verbose:
        logger.info(f"Parsing ratings response of length {len(text)}")
        logger.info(f"Raw response: {repr(text[:500])}")

    # The multiple passes are to for edge cases that emerge when using 8B or smaller models for generating synthetic data. This is to make a comprehensive parser for faster protoyping.
    # With 70B or bigger model, `json.load()` should "just work"
    try:
        # Handle the common case of indented JSON with newlines
        # First, remove any markdown or text before/after the JSON
        # Look for standard JSON start/end markers
        json_content = text.strip()

        # Try to normalize escape sequences
        json_content = json_content.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")

        # Check if we have a JSON object
        if "{" in json_content and "}" in json_content:
            start_idx = json_content.find("{")
            end_idx = json_content.rfind("}") + 1
            json_text = json_content[start_idx:end_idx]

            # Clean up the JSON string to handle common issues
            # First, convert newlines to spaces in JSON
            json_text = re.sub(r"\s*\n\s*", " ", json_text)

            # Now, try to parse it
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict) and (
                    "question" in parsed and "answer" in parsed and "rating" in parsed
                ):
                    parsed["rating"] = int(float(parsed["rating"]))
                    return [parsed]
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error for object: {str(e)}")

        # Check if we have a JSON object which is not completed
        if "{" in json_content and "}" not in json_content:
            start_idx = json_content.find("{")
            # Clean up the malformed format: ```json { .... ```
            end_idx = json_content.find("```")
            if end_idx <= 0:
                end_idx = len(json_content)
            # Clean up the JSON string to handle incompleted object
            if json_content[-1] == "\"":
                json_text = json_content[start_idx:end_idx] + "}"
            else:
                json_text = json_content[start_idx:end_idx] + "\"}"
            # Clean up the JSON string to handle common issues
            # First, convert newlines to spaces in JSON
            json_text = re.sub(r"\s*\n\s*", " ", json_text)

            # Now, try to parse it
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict) and (
                    "question" in parsed and "answer" in parsed and "rating" in parsed
                ):
                    parsed["rating"] = int(float(parsed["rating"]))
                    return [parsed]
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error for object: {str(e)}")

        # Check if we have a JSON array
        if "[" in json_content and "]" in json_content:
            start_idx = json_content.find("[")
            end_idx = json_content.rfind("]") + 1
            json_text = json_content[start_idx:end_idx]

            # Clean up the JSON string
            json_text = re.sub(r"\s*\n\s*", " ", json_text)

            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict) or "rating" not in item:
                            logger.debug(f"Array contains invalid item: {item}")
                            return []
                    logger.debug(f"Successfully parsed {len(parsed)} items in JSON array")
                    return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error for array: {str(e)}")

    except Exception as e:
        logger.debug(f"Error in primary parsing approach: {str(e)}")

    # Fallback to more specific methods
    # Method 1: Code block extraction
    try:
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_blocks:
            for block in code_blocks:
                try:
                    # Clean up newlines in the code block
                    clean_block = re.sub(r"\s*\n\s*", " ", block.strip())
                    parsed = json.loads(clean_block)
                    if isinstance(parsed, dict) and (
                        "question" in parsed and "answer" in parsed and "rating" in parsed
                    ):
                        logger.debug("Successfully parsed from code block (single object)")
                        return [parsed]
                    elif isinstance(parsed, list):
                        valid_items = True
                        for item in parsed:
                            if not isinstance(item, dict) or "rating" not in item:
                                valid_items = False
                                break
                        if valid_items and len(parsed) > 0:
                            if verbose:
                                logger.info(
                                    f"Successfully parsed {len(parsed)} items from code block"
                                )
                            return parsed
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        if verbose:
            logger.info(f"Error in code block extraction: {str(e)}")

    # Method 2: Regex
    try:
        # Look for JSON patterns in the text
        json_patterns = [
            # Single object pattern
            r'(\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"answer"\s*:\s*"[^"]*"\s*,\s*"rating"\s*:\s*\d+(?:\.\d+)?\s*\})',
            # Array pattern
            r'(\[\s*\{\s*"question"\s*:.*"rating"\s*:\s*\d+(?:\.\d+)?\s*\}\s*\])',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        # Clean up newlines in the match
                        clean_match = re.sub(r"\s*\n\s*", " ", match)
                        parsed = json.loads(clean_match)
                        if isinstance(parsed, dict) and (
                            "question" in parsed and "answer" in parsed and "rating" in parsed
                        ):
                            if verbose:
                                logger.info("Successfully parsed using regex (single object)")
                            return [parsed]
                        elif isinstance(parsed, list) and all("rating" in item for item in parsed):
                            if verbose:
                                logger.info(f"Successfully parsed {len(parsed)} items using regex")
                            return parsed
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        if verbose:
            logger.info(f"Error in regex extraction: {str(e)}")

    # Method 3: Try using json5 if available (more lenient parser)
    try:
        import json5

        try:
            parsed = json5.loads(text)
            if isinstance(parsed, dict) and (
                "question" in parsed and "answer" in parsed and "rating" in parsed
            ):
                if verbose:
                    logger.info("Successfully parsed using json5 (single object)")
                return [parsed]
            elif isinstance(parsed, list) and all("rating" in item for item in parsed):
                if verbose:
                    logger.info(f"Successfully parsed {len(parsed)} items using json5")
                return parsed
        except:
            pass
    except ImportError:
        if verbose:
            logger.info("json5 not available")

    # If we reach here, try one last aggressive approach
    try:
        # Try line-by-line parsing for each item
        if original_items and len(original_items) > 0:
            # Look for patterns that include both the question and rating
            found_items = []
            for item in original_items:
                # Escape regex special characters in question text
                question_escaped = re.escape(item.get("question", ""))
                pattern = f'.*{question_escaped}.*"rating"\\s*:\\s*(\\d+(?:\\.\\d+)?)'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    try:
                        rating = float(match.group(1))
                        found_items.append(
                            {
                                "question": item.get("question", ""),
                                "answer": item.get("answer", ""),
                                "rating": rating,
                            }
                        )
                        if verbose:
                            logger.info(
                                f"Found rating {rating} for question: {item.get('question', '')[:30]}..."
                            )
                    except:
                        pass

            if found_items:
                if verbose:
                    logger.info(f"Extracted {len(found_items)} ratings using pattern matching")
                return found_items
    except Exception as e:
        if verbose:
            logger.info(f"Error in final extraction attempt: {str(e)}")

    # If we reach here, we couldn't extract valid JSON
    if verbose:
        logger.info("All parsing methods failed")

    raise ValueError(f"Could not parse JSON with ratings: {text}")


def convert_to_conversation_format(
    qa_pairs: List[Dict[str, str]], system_prompt: Optional[str] = None
) -> List[List[Dict[str, str]]]:
    """Convert QA pairs to conversation format"""
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant that provides accurate, detailed responses."

    conversations = []
    for pair in qa_pairs:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ]
        conversations.append(conversation)

    return conversations
