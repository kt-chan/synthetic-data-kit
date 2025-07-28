# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Output utilities
import os
import chromadb
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

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

from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np


class RAGProccesor:
    def __init__(self, client: LLMClient, config_path: Optional[Path] = None):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client

        # Load config
        self.config = load_config(config_path)

        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)
        self.rag_host = get_rag_config(self.config).get("rag_host", "localhost")
        self.rag_port = get_rag_config(self.config).get("rag_port", 9000)
        self.rag_collection_name = get_rag_config(self.config).get("rag_collection_name", "default")
        self.rag_embedding_model = get_rag_config(self.config).get(
            "rag_model_name", "all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.rag_embedding_model)
        self.es_host = get_rag_config(self.config).get("es_host", "localhost")
        self.es_port = get_rag_config(self.config).get("es_port", 9200)
        self.es_index_name = get_rag_config(self.config).get("es_index_name", "default")
        self.es_user = get_rag_config(self.config).get("es_user", None)
        self.es_password = get_rag_config(self.config).get("es_password", None)
        self.es_mapping = {"properties": {"text": {"type": "text"}}}
        self.es_client = Elasticsearch(
            hosts=[{"host": self.es_host, "port": self.es_port, "scheme": "http"}],
            http_auth=(
                (self.es_user, self.es_password) if self.es_user and self.es_password else None
            ),
        )

    def truncate(self):
        self.get_collection(truncate=True)

    def get_collection(self, truncate: bool = False) -> chromadb.Collection:
        """
        # This is to initailize vector database for vector search
        """
        client = chromadb.HttpClient(host=self.rag_host, port=self.rag_port)
        collection_name = self.rag_collection_name

        # Create collection. get_collection, get_or_create_collection, delete_collection also available!
        if truncate:
            try:
                collection = client.delete_collection(name=collection_name)
                collection = client.create_collection(name=collection_name)

                # This is for BM2.5 on elastic search
                if self.es_client.indices.exists(index=self.es_index_name):
                    self.es_client.indices.delete(index=self.es_index_name)
                self.es_client.indices.create(index=self.es_index_name, mappings=self.es_mapping)

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

    def write_es_index(self, chunks: List[str]) -> bool:
        # Function to index documents
        try:
            actions = [
                {"_index": self.es_index_name, "_source": {"text": chunk}} for chunk in chunks
            ]
            bulk(self.es_client, actions)
            return True
        except Exception as e:
            print(f"\n ElasticSearch Error processing with exception:/n {str(e)}")
            return False

    def search_es_index(self, query, top_k=3) -> List[Tuple[str, float]]:
        query_body = {"query": {"match": {"text": query}}}
        response = self.es_client.search(index=self.es_index_name, body=query_body, size=top_k)
        return [(hit["_source"]["text"], hit["_score"]) for hit in response["hits"]["hits"]]

    def wrte_chunks(self, chunks: list[str], metas: list[dict], truncate: bool = False) -> bool:
        try:
            collection = self.get_collection(truncate)
            ids = [str(i) for i in range(len(chunks))]
            embeddings = self.model.encode(chunks)
            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
            self.write_es_index(chunks)
            print(f"Loaded {str(collection.count())} chunks into {self.rag_collection_name}")
            return True
        except Exception as e:
            print(f"\n VectorDB Error processing with exception:/n {str(e)}")
            return False

    def query(self, question: str, answer: str = None, topK: int = 3) -> Dict[str, str]:
        try:
            # Get the collection
            collection = self.get_collection()

            # Perform the query using the vector database
            results = collection.query(query_embeddings=self.model.encode([question]), n_results=topK)

            # Extract documents and summaries  and their embeddings
            documents = results["documents"][0]
            summaries = [result['summary'] for result in results['metadatas'][0] if 'summary' in result]
            corpus_embeddings = self.model.encode(summaries, convert_to_tensor=True)

            # Encode the query to a vector
            query_embedding = self.model.encode([question], convert_to_tensor=True)

            # Compute cosine similarity scores
            cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].numpy()

            # Perform BM25 search using Elasticsearch
            bm25_results = self.search_es_index(question)

            # Extract document texts and BM25 scores
            documents_bm25, bm25_scores_list = zip(*bm25_results)
            bm25_scores = np.array(bm25_scores_list)

            # Normalize BM25 and cosine similarity scores
            bm25_scores_normalized = (
                bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
            )
            cosine_scores_normalized = (
                cosine_scores / np.max(cosine_scores)
                if np.max(cosine_scores) > 0
                else cosine_scores
            )

            # Combine BM25 and cosine similarity scores using a weighted approach
            alpha = 0.5  # Weight for BM25, (1 - alpha) for dense search
            hybrid_scores = alpha * bm25_scores_normalized + (1 - alpha) * cosine_scores_normalized

            # Rank results based on hybrid scores
            ranked_results = sorted(zip(documents, hybrid_scores), key=lambda x: x[1], reverse=True)

            # Extract the top result
            top_results = ranked_results[topK]

            # Print the question and the top result
            print(
                f"Question: {question}; Answer: {top_results[0][0]} (Hybrid Score: {top_results[0][1]:.4f})"
            )

            # Return the top result as a dictionary
            # @TODO Fetch Context And then push to LLM to generate better answer
            return {
                "question": question,
                "answer": top_results[0][0],
                "hybrid_score": str(top_results[0][1]),
            }

        except Exception as e:
            print(f"Error processing with exception:\n {str(e)}")
            return {"error": str(e)}
