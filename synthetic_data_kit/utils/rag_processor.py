import os
import chromadb
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from concurrent.futures import ThreadPoolExecutor, as_completed

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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.connections = get_rag_config(self.config).get("connections", 4)
        self.threads = get_rag_config(self.config).get("threads", 4)
        self.retries = get_rag_config(self.config).get("max_retries", 3)
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
            logger.info(f"Truncating datastores .... ")
            try:
                collection = client.delete_collection(name=collection_name)
                collection = client.create_collection(name=collection_name)
                logger.info(f"Truncated vector datastore. ")
                # This is for BM2.5 on elastic search
                if self.es_client.indices.exists(index=self.es_index_name):
                    self.es_client.indices.delete(index=self.es_index_name)
                    logger.info(f"Truncated fulltext search datastore. ")
                self.es_client.indices.create(index=self.es_index_name, mappings=self.es_mapping)

            except Exception as e:
                # Collection does not exist, create it
                collection = client.create_collection(name=collection_name)
        else:
            try:
                collection = client.get_or_create_collection(name=collection_name)
            except Exception as e:
                # Collection does not exist, create it
                logger.error(f"Could not create vectordatabase collection:\n {str(e)}")
                raise Exception(f"Could not create vectordatabase collection:\n {str(e)}")

        return collection

    def write_es_index(self, chunks: List[str]):
        # Function to index documents
        try:
            actions = [
                {"_index": self.es_index_name, "_source": {"text": chunk}} for chunk in chunks
            ]
            bulk(self.es_client, actions)
            return True
        except Exception as e:
            logger.error(f"\n ElasticSearch Error processing with exception:/n {str(e)}")
            raise Exception(f"\n ElasticSearch Error processing with exception:/n {str(e)}")

    def search_es_index(self, query, top_k=3) -> List[Tuple[str, float]]:
        query_body = {"query": {"match": {"text": query}}}
        response = self.es_client.search(index=self.es_index_name, body=query_body, size=top_k)
        return [(hit["_source"]["text"], hit["_score"]) for hit in response["hits"]["hits"]]

    def wrte_chunks(self, chunks: list[str], metas: list[dict], truncate: bool = False):
        try:
            collection = self.get_collection(truncate)
            ids = [str(i) for i in range(len(chunks))]
            embeddings = self.model.encode(chunks)
            if len(embeddings) == 0:
                raise Exception(f"\n VectorDB Error processing with empty embeddings")
            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
            self.write_es_index(chunks)
            logger.info(f"Loaded {str(collection.count())} chunks into {self.rag_collection_name}")
        except Exception as e:
            logger.error(f"\n VectorDB Error processing with exception:/n {str(e)}")
            raise Exception(f"\n VectorDB Error processing with exception:/n {str(e)}")

    def _process_single_qa(self, qa_pair: Dict[str, str], topK: int) -> List[Dict[str, str]]:
        """Process a single QA pair with retries and return enriched message"""
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        retry_count = 0
        message:List[Dict[str, str]] = None

        while retry_count < self.retries and message is None:
            try:
                collection = self.get_collection()
                results = collection.query(
                    query_embeddings=self.model.encode([question]), n_results=topK
                )

                # Extract documents and summaries and their embeddings
                vector_documents = results["documents"][0]
                vector_summaries = [
                    result["summary"] for result in results["metadatas"][0] if "summary" in result
                ]
                corpus_embeddings = self.model.encode(vector_summaries, convert_to_tensor=True)

                # Encode the query to a vector
                query_embedding = self.model.encode([question], convert_to_tensor=True)

                # Compute cosine similarity scores
                vector_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].numpy()

                # Perform BM25 search using Elasticsearch
                bm25_results = self.search_es_index(question, topK)

                # Extract document texts and BM25 scores
                bm25_documents, bm25_scores_list = zip(*bm25_results)
                bm25_scores = np.array(bm25_scores_list)

                # Normalize BM25 and cosine similarity scores
                bm25_scores_normalized = (
                    bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
                )
                vector_scores_normalized = (
                    vector_scores / np.max(vector_scores)
                    if np.max(vector_scores) > 0
                    else vector_scores
                )

                # Ensure both normalized arrays have the same length
                min_length = min(len(bm25_scores_normalized), len(vector_scores_normalized))
                bm25_scores_normalized = bm25_scores_normalized[:min_length]
                vector_scores_normalized = vector_scores_normalized[:min_length]

                # Combine BM25 and cosine similarity scores using a weighted approach
                alpha = 0.5  # Weight for BM25, (1 - alpha) for dense search
                hybrid_scores = (
                    alpha * bm25_scores_normalized + (1 - alpha) * vector_scores_normalized
                )

                # Rank results based on hybrid scores
                top_results = sorted(
                    zip(vector_summaries, vector_documents, bm25_documents, hybrid_scores),
                    key=lambda x: x[3],
                    reverse=True,
                )

                if top_results and len(top_results) > 0:
                    rag_question = str(question).strip()
                    rag_answer = str(answer).strip()
                    rag_content_summary = str(top_results[0][0]).strip()
                    rag_content_detail = (
                        str(top_results[0][1]).strip() + " \n " + str(top_results[0][2]).strip()
                    )

                    qa_enrichment_prompt_template = get_prompt(self.config, "qa_enrichment")
                    qa_enrichment_prompt = qa_enrichment_prompt_template.format(
                        question=rag_question,
                        answer=rag_answer,
                        chunk_summary=rag_content_summary,
                        chunk_text=rag_content_detail,
                    ).strip()
                    message = [{"role": "system", "content": qa_enrichment_prompt}]
                    break  # Success, break retry loop
                else:
                    raise Exception("No results found for query")

            except Exception as e:
                retry_count += 1
                if retry_count >= self.retries:
                    logger.error(
                        f"Failed to process QA pair after {self.retries} retries: {question} / {answer}"
                    )
                    raise Exception(
                        f"Failed to process ragClient.query after maximum retries for qa pair: {question} / {answer}"
                    )
                else:
                    logger.warning(
                        f"Retry {retry_count}/{self.retries} for QA pair due to error: {str(e)}"
                    )

        return message

    def query(self, qa_pairs: List[Dict[str, str]], topK: int = 3) -> List[List[Dict[str, str]]]:
        """Process QA pairs concurrently using thread pool"""
        output_list:List[List[Dict[str, str]]] = []
        with ThreadPoolExecutor(max_workers=min(self.connections, len(qa_pairs))) as executor:
            futures = {executor.submit(self._process_single_qa, qa, topK): qa for qa in qa_pairs}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    output_list.append(result)
                except Exception as e:
                    # Cancel all pending futures on first error
                    for f in futures:
                        f.cancel()
                    logger.error(f"Processing failed: {str(e)}")
                    raise  # Re-raise exception to stop entire process

        return output_list
