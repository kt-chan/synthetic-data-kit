"""Unit tests for document parsers."""

import pytest
from pymilvus import MilvusClient
from pymilvus import model


@pytest.mark.unit
def test_Milvus_setup():
    """Test RAG Setup."""
    client = MilvusClient(uri="http://ubuntu.wsl.local:19530", token="root:Milvus")
    client.list_databases()
    assert len(client.list_databases()) == 1


@pytest.mark.unit
def test_rag_setup():
    client = MilvusClient(uri="http://ubuntu.wsl.local:19530", token="root:Milvus")

    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")
        client.create_collection(
            collection_name="demo_collection",
            dimension=768,  # The vectors we will use in this demo has 768 dimensions
        )

    assert len(client.list_collections()) == 1

    embedding_fn = model.DefaultEmbeddingFunction()

    # Text strings to search from.
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]

    vectors = embedding_fn.encode_documents(docs)
    # The output vector has 768 dimensions, matching the collection that we just created.
    print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

    # Each entity has id, vector representation, raw text, and a subject label that we use
    # to demo metadata filtering later.
    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
        for i in range(len(vectors))
    ]

    print("Data has", len(data), "entities, each with fields: ", data[0].keys())
    print("Vector dim:", len(data[0]["vector"]))
