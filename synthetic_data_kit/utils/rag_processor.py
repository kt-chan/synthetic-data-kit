# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Output utilities
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def get_collection(collection_name: str, truncate: bool = False) -> chromadb.Collection:
    client = chromadb.HttpClient(host="ubuntu.wsl.local", port=9000)
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


def reset_collection(collection_name: str = "synthetic_data_kit"):
    get_collection(collection_name=collection_name, truncate=True)


def wrte_chunks(
    chunks: list[str], metas: list[dict], collection_name: str = "synthetic_data_kit"
) -> bool:
    try:
        collection = get_collection(collection_name)
        ids = [str(i) for i in range(len(chunks))]
        collection.add(ids=ids, documents=chunks, metadatas=metas)
        print(f"Loaded {str(collection.count())} chunks into {collection_name}")
        return True
    except Exception as e:
        print(f"  Error processing with exception:/n {str(e)}")
        return False
