# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Text processing utilities
import re
import json
from typing import List, Dict, Any

# def split_into_chunks(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
#     """Split text into chunks with optional overlap"""
#     paragraphs = text.split("\n\n")
#     chunks = []
#     current_chunk = ""

#     for para in paragraphs:
#         if len(current_chunk) + len(para) > chunk_size and current_chunk:
#             chunks.append(current_chunk)
#             # Keep some overlap for context
#             sentences = current_chunk.split('. ')
#             if len(sentences) > 3:
#                 current_chunk = '. '.join(sentences[-3:]) + "\n\n" + para
#             else:
#                 current_chunk = para
#         else:
#             if current_chunk:
#                 current_chunk += "\n\n" + para
#             else:
#                 current_chunk = para

#     if current_chunk:
#         chunks.append(current_chunk)

#     return chunks


def split_into_chunks(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split the input text into chunks of at most `chunk_size` characters,
    ensuring that each successive chunk overlaps the previous by
    approximately `overlap` characters.

    The function tries to break at paragraph boundaries first, then at
    sentence boundaries, and finally at word boundaries if necessary.
    """

    chunks: List[str] = []

    # Edge-case: if the text is already shorter than the chunk size
    if len(text) <= chunk_size:
        return [text]

    start = 0
    split_at = 0
    text_len = len(text)

    while start < text_len:
        # Proposed end for this chunk
        end = start + chunk_size

        if end >= text_len:
            # Last chunk: just take the rest
            chunks.append(text[start:])
            break

        # 1. Look back for a paragraph break
        para_break = text.rfind("\n\n", start, end)
        if para_break != -1 and split_at < para_break +2 & para_break + 2 < end:
            split_at = para_break + 2
        else:
            # 2. Look back for a sentence break
            sent_break = text.rfind(". ", start, end)
            if sent_break != -1 and sent_break + 2 < end:
                split_at = sent_break + 2
            else:
                # 3. Find the last space within the chunk
                space_break = text.rfind(" ", start, end)
                if space_break != -1:
                    split_at = space_break + 1
                else:
                    # Fallback: hard cut at chunk_size
                    split_at = end

        chunks.append(text[start:split_at])
        # Move the start pointer back by `overlap`, but never before 0
        start = max(split_at - overlap, 0)

        # Prevent infinite loops if the overlap is very large or chunk is tiny
        if start + overlap > split_at and start > 0:
            start = split_at

    return chunks


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text that might contain markdown or other content"""
    text = text.strip()

    # Try to parse as complete JSON
    if text.startswith("{") and text.endswith("}") or text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Look for JSON within Markdown code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try a more aggressive pattern
    json_pattern = r"\{[\s\S]*\}|\[[\s\S]*\]"
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from the response")
