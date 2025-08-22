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
from synthetic_data_kit.utils.app_logger import get_logger

logger = get_logger(__name__)


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
        self.threads = get_rag_config(self.config).get("threads", 4)
        self.top_n = get_rag_config(self.config).get("top_n", 3)
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

    def _write_to_vector(self, chunks: list[str], metas: list[dict]) -> bool:
        try:
            collection = self.get_collection()
            ids = [str(i) for i in range(len(chunks))]
            embeddings = self.model.encode(chunks)
            if len(embeddings) == 0:
                raise Exception(f"VectorDB Error processing with empty embeddings")

            ## Delete first for overwrite the collection with the same filenames
            filesnames = list({meta_item["filename"] for meta_item in metas})
            collection.delete(where={"filename": {"$in": filesnames}})

            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
            logger.info(
                f"Loaded {str(collection.count())} chunks into {self.rag_collection_name} and {self.es_index_name}"
            )
            return True
        except Exception as e:
            logger.error(f"\n VectorDB store Error processing with exception:/n {str(e)}")
            return False

    def _write_to_text(self, chunks: list[str], metas: list[dict]) -> bool:
        # Function to index documents

        try:
            filenames, chunkids = zip(
                *[(meta_item["filename"], meta_item["chunkid"]) for meta_item in metas]
            )

            delete_query = {
                "query": {
                    "terms": {
                        "filename.keyword": list(
                            set(filenames)
                        )  # Use .keyword if filename is not analyzed
                    }
                }
            }
            delete_response = self.es_client.delete_by_query(
                index=self.es_index_name, body=delete_query
            )

            if len(delete_response.get("failures")) == 0 and delete_response.get("deleted", 0) > 0:
                logger.debug(f"Success: Deleted {delete_response['deleted']} documents in fulltext search store.")

            insert_actions = [
                {
                    "_index": self.es_index_name,
                    "_source": {"filename": filename, "chunkid": chunkid, "text": chunk},
                }
                for filename, chunkid, chunk in zip(filenames, chunkids, chunks)
            ]
            bulk(self.es_client, insert_actions)
            return True
        except Exception as e:
            logger.error(f"\n Fulltext search store Error processing with exception:/n {str(e)}")
            return False

    def search_es_index(self, query, top_k=3) -> List[Tuple[str, float]]:
        query_body = {"query": {"match": {"text": query}}}
        response = self.es_client.search(index=self.es_index_name, body=query_body, size=top_k)
        return [(hit["_source"]["text"], hit["_score"]) for hit in response["hits"]["hits"]]

    def wrte_chunks(self, chunks: list[str], metas: list[dict]):
        """
        metas data:
        "filename": item["filename"],
        "chunkid": item["filename"] + "_" + str(item["id"]),
        "summary": item["summary"],
        """
        try:
            vector_operation = self._write_to_vector(chunks, metas)
            fulltext_operation = self._write_to_text(chunks, metas)
            return vector_operation and fulltext_operation
        except Exception as e:
            raise Exception(f"Chunk writing error processing with exception:/n {str(e)}")

    def _process_single_qa_prompt(
        self, qa_pair: Dict[str, str], max_chars: int = 10000
    ) -> List[Dict[str, str]]:
        """Process a single QA pair with retries and return enriched message"""
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        top_n = self.top_n
        message: List[Dict[str, str]] = None

        retry_count = 0
        while retry_count < self.retries and message is None:
            try:
                collection = self.get_collection()
                results = collection.query(
                    query_embeddings=self.model.encode([question]), n_results=top_n
                )

                # Extract documents and summaries and their embeddings
                vector_documents = results["documents"][0]
                vector_summaries = [
                    result["summary"] for result in results["metadatas"][0] if "summary" in result
                ]
                vector_filenames = [
                    result["filename"] for result in results["metadatas"][0] if "filename" in result
                ]
                corpus_embeddings = self.model.encode(vector_summaries, convert_to_tensor=True)

                # Encode the query to a vector
                query_embedding = self.model.encode([question], convert_to_tensor=True)

                # Compute cosine similarity scores
                vector_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].numpy()

                # Perform BM25 search using Elasticsearch
                bm25_results = self.search_es_index(question, top_n)

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

                # # Debug, instead of hybrid score, just append the best matched records on both
                # # Combine BM25 and cosine similarity scores using a weighted approach
                # alpha = 0.5  # Weight for BM25, (1 - alpha) for dense search
                # hybrid_scores = (
                #     alpha * bm25_scores_normalized + (1 - alpha) * vector_scores_normalized
                # )

                # # Rank results based on hybrid scores
                # top_results = sorted(
                #     zip(vector_summaries, vector_documents, bm25_documents, hybrid_scores),
                #     key=lambda x: x[3],
                #     reverse=True,
                # )

                # if top_results and len(top_results) > 0:
                if (min_length is not None) and (min_length > 0):
                    rag_question = str(question).strip()
                    rag_answer = str(answer).strip()

                    metadata: List[str] = []

                    for idx in range(min(top_n, min_length)):
                        contents = []
                        contents.append(f"Reference {idx+1}")
                        contents.append(f"Source Filepath: {vector_filenames[idx].strip()}")
                        contents.append(f"Reference Content: ")
                        contents.append(str(vector_summaries[idx]).strip())
                        if str(vector_documents[idx]).strip() == str(bm25_documents[idx]).strip():
                            contents.append(str(vector_documents[idx]).strip())
                        else:
                            contents.append(str(vector_documents[idx]).strip())
                            contents.append(str(bm25_documents[idx]).strip())
                        metadata.append("\n".join(contents).strip())

                    qa_enrichment_prompt_template = get_prompt(self.config, "qa_enrichment")
                    chunk_text = "\n".join(metadata).strip()
                    qa_enrichment_prompt = qa_enrichment_prompt_template.format(
                        question=rag_question,
                        answer=rag_answer,
                        chunk_text=chunk_text[
                            : int(min(len(chunk_text), max(2000, int(max_chars / 2))))
                        ],
                    ).strip()
                    message = [{"role": "system", "content": qa_enrichment_prompt}]
                    break  # Success, break retry loop
                else:
                    raise Exception("No results found for query")

            except Exception as e:
                retry_count += 1
                logger.error(f"failed at retries: {retry_count}, with error: {e}")
                if retry_count >= self.retries:
                    logger.error(
                        f"Failed to process QA pair after {self.retries} retries with Q/A Pair as: {question} / {answer}"
                    )
                    raise Exception(
                        f"Failed to process ragClient.query after maximum retries for Q/A Pair as : {question} / {answer}"
                    )
                else:
                    logger.warning(
                        f"Retry {retry_count}/{self.retries} for QA pair due to error: {str(e)}"
                    )

        return message

    def _process_single_qa_content(
        self, qa_pair: Dict[str, str], max_chars: int = 10000
    ) -> Dict[str, str]:
        """Process a single QA pair with retries and return enriched message"""
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        top_n = self.top_n
        output: Dict[str, str] = None

        retry_count = 0
        while retry_count < self.retries and output is None:
            try:
                collection = self.get_collection()
                logger.debug(f"vectordb size: {collection.count()}")
                search_text = ", ".join([question, answer])
                results = collection.query(
                    query_embeddings=self.model.encode(search_text), n_results=top_n
                )
                logger.debug(f"results size: {len(results)}")
                # Extract documents and summaries and their embeddings
                vector_documents = results["documents"][0]
                vector_summaries = [
                    result["summary"] for result in results["metadatas"][0] if "summary" in result
                ]
                vector_filenames = [
                    result["filename"] for result in results["metadatas"][0] if "filename" in result
                ]
                corpus_embeddings = self.model.encode(vector_summaries, convert_to_tensor=True)

                # Encode the query to a vector
                query_embedding = self.model.encode([question, answer], convert_to_tensor=True)

                # Compute cosine similarity scores
                vector_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].numpy()

                # Perform BM25 search using Elasticsearch
                bm25_results = self.search_es_index(search_text, top_n)

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

                # # Debug, instead of hybrid score, just append the best matched records on both
                # # Combine BM25 and cosine similarity scores using a weighted approach
                # alpha = 0.5  # Weight for BM25, (1 - alpha) for dense search
                # hybrid_scores = (
                #     alpha * bm25_scores_normalized + (1 - alpha) * vector_scores_normalized
                # )

                # # Rank results based on hybrid scores
                # top_results = sorted(
                #     zip(vector_summaries, vector_documents, bm25_documents, hybrid_scores),
                #     key=lambda x: x[3],
                #     reverse=True,
                # )

                # if top_results and len(top_results) > 0:
                if (min_length is not None) and (min_length > 0):
                    rag_question = str(question).strip()
                    rag_answer = str(answer).strip()

                    metadata: List[str] = []

                    for idx in range(min(top_n, min_length)):
                        contents = []
                        contents.append(f"Reference Content {idx+1}:")
                        contents.append(str(vector_summaries[idx]).strip())
                        if str(vector_documents[idx]).strip() == str(bm25_documents[idx]).strip():
                            contents.append(str(vector_documents[idx]).strip())
                        else:
                            contents.append(str(vector_documents[idx]).strip())
                            contents.append(str(bm25_documents[idx]).strip())
                        metadata.append("\n".join(contents).strip())

                    chunk_text = "\n".join(metadata).strip()
                    output = {
                        "question": rag_question,
                        "answer": rag_answer,
                        "content": chunk_text[
                            : int(min(len(chunk_text), max(2000, int(max_chars / 2))))
                        ].strip(),
                    }
                    break  # Success, break retry loop
                else:
                    raise Exception("No results found for query")

            except Exception as e:
                retry_count += 1
                logger.error(f"failed at retries: {retry_count}, with error: {e}")
                if retry_count >= self.retries:
                    logger.error(
                        f"Failed to process QA pair after {self.retries} retries with Q/A Pair as: {question} / {answer}"
                    )
                    raise Exception(
                        f"Failed to process ragClient.query after maximum retries for Q/A Pair as : {question} / {answer}"
                    )
                else:
                    logger.warning(
                        f"Retry {retry_count}/{self.retries} for QA pair due to error: {str(e)}"
                    )

        return output

    def buildPrompt(
        self, qa_pairs: List[Dict[str, str]], max_chars: int = 4000
    ) -> List[List[Dict[str, str]]]:
        """Process QA pairs concurrently using thread pool"""
        output_list: List[List[Dict[str, str]]] = []
        try:
            with ThreadPoolExecutor(max_workers=min(self.threads, len(qa_pairs))) as executor:
                futures = {
                    executor.submit(self._process_single_qa_prompt, qa, max_chars=max_chars)
                    for qa in qa_pairs
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        output_list.append(result)
                    except Exception as e:
                        # # Cancel all pending futures on first error
                        # for f in futures:
                        #     f.cancel()
                        logger.error(f"Processing failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error in buildPrompt: {e}")

        return output_list

    def enrichQAPair(
        self, qa_pairs: List[List[Dict[str, str]]], max_chars: int = 10000
    ) -> List[Dict[str, str]]:
        # output_list = []

        # for idx1, qa_pair_list in enumerate(qa_pairs):
        #     for idx2, qa_pair in enumerate(qa_pair_list):
        #         response = self._process_single_qa(qa_pair, max_chars)
        #         if response is not None and len(response) == 3:
        #             output_list.append(response)

        output_list: List[Dict[str, str]] = []
        try:
            with ThreadPoolExecutor(max_workers=min(self.threads, len(qa_pairs))) as executor:
                futures = {
                    executor.submit(self._process_single_qa_content, qa, max_chars=max_chars)
                    for qa in qa_pairs
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        output_list.append(result)
                    except Exception as e:
                        # Cancel all pending futures on first error
                        # for f in futures:
                        #     f.cancel()
                        logger.error(f"Processing failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error in buildPrompt: {e}")

        return output_list
