"""Unit tests for document parsers."""

import pytest
import chromadb
import uuid

# 连接 docker-compose 里的 chroma 服务


@pytest.mark.unit
def test_ragdb_setup():
    """Test RAG Setup."""
    client = chromadb.HttpClient(host="ubuntu.wsl.local", port=9000)
    # 心跳检测
    heartbeat = client.heartbeat()
    print("heartbeat:", heartbeat)
    assert heartbeat > 1


@pytest.mark.unit
def test_ragdb_access():
    client = chromadb.HttpClient(host="ubuntu.wsl.local", port=9000)

    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    try:
        collection = client.delete_collection(name="all-my-documents")
        collection = client.create_collection(name="all-my-documents")
    except Exception as e:
        # Collection does not exist, create it
        collection = client.create_collection(name="all-my-documents")

    # Add docs to the collection. Can also update and delete. Row-based API coming soon!
    collection.add(
        documents=[
            "I love docker-compose",
            "Chroma is a vector DB",
        ],  # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadatas=[{"source": "notion"}, {"source": "google-docs"}],  # filter on these!
        ids=["doc1", "doc2"],  # unique for each doc
    )

    # Query/search 2 most similar results. You can also .get by id
    results = collection.query(
        query_texts=["how to use chroma"],
        n_results=1,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )
    print("Top result:", results["documents"][0][0])

    assert len(results["documents"]) >= 1 and results["documents"][0][0].strip() == "Chroma is a vector DB"

@pytest.mark.unit
def test_ragdb_query():
    client = chromadb.HttpClient(host="ubuntu.wsl.local", port=9000)
    collection = client.get_or_create_collection(name="synthetic_data_kit")
    results = collection.query(query_texts=["what is last stand for?"],n_results=3, )
    print("Top result:", results["documents"][0][0])
    