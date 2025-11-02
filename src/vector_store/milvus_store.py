"""
Milvus Vector Store Module
--------------------------
Handles:
1. Connection setup to Milvus.
2. Schema creation (for legal RAG chunks).
3. Insertion of embedding vectors + metadata.
4. Query interface for retrieval.
"""

from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any
from src.config.settings import settings


# ===============================================================
# 1️⃣ Milvus Connection Setup
# ===============================================================
def connect_milvus():
    """
    Connects to Milvus instance using MilvusClient.
    """
    try:
        client = MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        )
        print(f"✅ Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        return client
    except Exception as e:
        raise ConnectionError(f"❌ Failed to connect to Milvus: {str(e)}")


# ===============================================================
# 2️⃣ Create Collection (Schema)
# ===============================================================
def create_collection(client: MilvusClient, collection_name: str = settings.MILVUS_COLLECTION):
    """
    Creates a collection with index parameters.
    """
    try:
        # Delete existing collection if it exists
        if client.has_collection(collection_name=collection_name):
            print(f"⚠️ Collection '{collection_name}' exists without index. Recreating...")
            delete_collection(client, collection_name)

        # Define schema
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=384)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        # Prepare index parameters
        index_params = client.prepare_index_params()
        
        index_params.add_index(
            field_name="id",
            index_type="AUTOINDEX"
        )
        
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # Create collection with index - this automatically loads it
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        print(f"✅ Created collection '{collection_name}' with index and auto-loaded")

    except Exception as e:
        raise RuntimeError(f"❌ Error creating collection: {str(e)}")


# ===============================================================
# 3️⃣ Insert Data into Milvus
# ===============================================================
def insert_chunks(client: MilvusClient, collection_name: str, chunks: List[Dict[str, Any]]):
    """
    Inserts chunks and their embeddings into Milvus.
    """
    try:
        data = [
            {
                "embedding": chunk["embedding"],
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            }
            for chunk in chunks
        ]

        client.insert(
            collection_name=collection_name,
            data=data
        )

        print(f"✅ Inserted {len(chunks)} chunks into Milvus collection '{collection_name}'.")
        return True

    except Exception as e:
        raise RuntimeError(f"❌ Error inserting data into Milvus: {str(e)}")


# ===============================================================
# 4️⃣ Load Collection (Required Before Search)
# ===============================================================
def load_collection(client: MilvusClient, collection_name: str):
    """
    Loads the collection into memory for searching.
    """
    try:
        client.load_collection(collection_name=collection_name)
        print(f"✅ Loaded collection '{collection_name}' into memory.")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading collection: {str(e)}")
    

# ===============================================================
# 🗑️ Delete Collection (for cleanup)
# ===============================================================
def delete_collection(client: MilvusClient, collection_name: str):
    """
    Drops a collection if it exists.
    """
    try:
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
            print(f"🗑️ Deleted collection '{collection_name}'")
        else:
            print(f"ℹ️ Collection '{collection_name}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"❌ Error deleting collection: {str(e)}")


# ===============================================================
# 5️⃣ Search Function (for Retrieval)
# ===============================================================
def search_similar_chunks(client: MilvusClient, collection_name: str, query_embedding: List[float], top_k: int = 3):
    """
    Searches for similar chunks in Milvus using vector similarity.
    """
    try:
        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["content", "metadata"]
        )

        matches = []
        for hit in results[0]:
            matches.append({
                "score": hit.get("distance"),
                "content": hit.get("entity", {}).get("content"),
                "metadata": hit.get("entity", {}).get("metadata")
            })

        return matches

    except Exception as e:
        raise RuntimeError(f"❌ Error during search: {str(e)}")