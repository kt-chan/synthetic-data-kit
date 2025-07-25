# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Output utilities
import os
import chromadb
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import (
    load_config,
    get_generation_config,
    get_curate_config,
    get_rag_config,
    get_prompt,
)


class RAGProccesor:
    def __init__(self, client: LLMClient, config_path: Optional[Path] = None):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client

        # Load config
        self.config = load_config(config_path)

        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)        
        self.rag_host = get_rag_config(self.config).get("host", "localhost")
        self.rag_port =  get_rag_config(self.config).get("port", 9000)
        self.rag_collection_name = get_rag_config(self.config).get("collection_name", "default")

    def get_collection(self, truncate: bool = False) -> chromadb.Collection:
        client = chromadb.HttpClient(host=self.rag_host, port=self.rag_port)
        collection_name = self.rag_collection_name

        # Create collection. get_collection, get_or_create_collection, delete_collection also available!
        if truncate:
            try:
                collection = client.delete_collection(name=collection_name)
                collection = client.create_collection(name=collection_name)
            except Exception as e:
                # Collection does not exist, create it
                collection = client.create_collection(name=collection_name)
        else:
            try:
                collection = client.get_or_create_collection(name=collection_name)
            except Exception as e:
                # Collection does not exist, create it
                raise ValueError(f"Could not create vectordatabase collection:\n {str(e)}")

        return collection


    def wrte_chunks(
        self, chunks: list[str], metas: list[dict], truncate: bool = False
    ) -> bool:
        try:
            collection = self.get_collection(truncate)
            ids = [str(i) for i in range(len(chunks))]
            collection.add(ids=ids, documents=chunks, metadatas=metas)
            print(f"Loaded {str(collection.count())} chunks into {self.rag_collection_name}")
            return True
        except Exception as e:
            print(f"  Error processing with exception:/n {str(e)}")
            return False

    def query(
            self, question: str, answer: str
    ) -> Dict[str, str]:
        try:
            collection = self.get_collection()
            results = collection.query(query_texts=[question],n_results=3)
            print(f"Question: {question}; Answer: {results["metadatas"][0][0]["summary"]}")
            return True
        except Exception as e:
            print(f"  Error processing with exception:/n {str(e)}")
            return False