"""
Milvus Ingestion Script (Per Category)
--------------------------------------
Loads pre-chunked JSON data, generates embeddings,
and inserts into Milvus collection for that category.
"""

import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.vector_store.milvus_store import connect_milvus, create_collection, insert_chunks, delete_collection
from src.config.settings import settings


# ===============================================================
# 1️⃣ Configuration
# ===============================================================
CHUNK_FOLDER = "./data/chunked_categories"   # your folder path
EMBEDDING_DIM = 768


# ===============================================================
# 2️⃣ Load Embedding Model
# ===============================================================
print("🔹 Loading embedding model (all-mpnet-base-v2)...")
embedder = SentenceTransformer("all-mpnet-base-v2")
print("✅ Model loaded!")


# ===============================================================
# 3️⃣ Milvus Connection
# ===============================================================
client = connect_milvus()


# ===============================================================
# 4️⃣ Helper Function: Process One Category
# ===============================================================
def process_category(category_file: str):
    category_name = os.path.splitext(os.path.basename(category_file))[0]
    collection_name = f"{category_name}_store"  # e.g. criminal_law_store
    if client.has_collection(collection_name=collection_name):
        print(f"⚠️ Collection '{collection_name}' exists without index. Recreating...")
        delete_collection(client, collection_name)

    print(f"\n📂 Processing category: {category_name}")
    print(f"📘 Creating / resetting collection: {collection_name}")

    # Create (or recreate) collection for this category
    create_collection(client, collection_name=collection_name)

    # Load chunks
    with open(category_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"🔹 Loaded {len(chunks)} chunks for {category_name}")

    # Prepare and insert in batches (avoid memory pressure)
    batch_size = 256
    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Embedding {category_name}"):
        batch = chunks[i:i + batch_size]

        # Generate embeddings for this batch
        texts = [chunk["content"] for chunk in batch]
        embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

        # Attach embeddings to chunks
        for j, chunk in enumerate(batch):
            chunk["embedding"] = embeddings[j]

        # Insert batch into Milvus
        insert_chunks(client, collection_name, batch)

    print(f"✅ Completed ingestion for {category_name}")


# ===============================================================
# 5️⃣ Run for All JSON Files in Folder
# ===============================================================
if __name__ == "__main__":
    files = [os.path.join(CHUNK_FOLDER, f) for f in os.listdir(CHUNK_FOLDER) if f.endswith(".json")]
    print(f"📁 Found {len(files)} category files to ingest.")

    for file_path in files:
        process_category(file_path)

    print("\n🎯 All categories ingested successfully into Milvus!")
