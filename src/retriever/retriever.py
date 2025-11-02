"""
Retriever Module
-----------------
Handles:
1. Loading the IPC JSON legal dataset.
2. Chunking long legal sections & sub-sections hierarchically.
3. Generating embeddings for each chunk (for Milvus insertion later).

This module prepares data for the RAG pipeline in LangGraph.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.config.settings import settings
from src.chunking.chunking import create_chunks


# ===============================================================
#  Data Loader
# ===============================================================
def load_legal_data(filepath: Path) -> List[Dict[str, Any]]:
    """
    Loads the IPC legal dataset from JSON file.

    Returns:
        list: Parsed list of chapters, sections, and sub-sections
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Dataset root element must be a list.")
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found at: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON format: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected error while loading data: {str(e)}")


# ===============================================================
# 2 Embedding Generator
# ===============================================================   
        
model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates embeddings for all chunks.
    Returns list with 'embedding' added to each dictionary.
    """
    try:
        texts = [chunk["content"] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
        return chunks
    except Exception as e:
        raise RuntimeError(f"❌ Error during embedding generation: {str(e)}")
    

def get_embedding(text: str) -> list:
    """
    Generates embedding for the given text.

    Args:
        text (str): The input text (IPC chunk or user query)
    Returns:
        list: Normalized embedding vector (length 384)
    """
    try:
        if not text:
            raise ValueError("Input text is empty")

        # Generate embedding as NumPy array
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    except Exception as e:
        raise RuntimeError(f"❌ Embedding generation failed: {str(e)}")


# ===============================================================
# 3 Utility Function — Complete Retriever Pipeline
# ===============================================================
def build_retriever_pipeline() -> List[Dict[str, Any]]:
    """
    Full pipeline:
    1. Load data
    2. Create chunks
    3. Generate embeddings
    """
    data_path = settings.DATA_DIR / "indian_penal_code.json"

    print("📘 Loading legal data...")
    data = load_legal_data(data_path)

    print("✂️ Creating chunks...")
    chunks = create_chunks(data, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    print(f"✅ Created {len(chunks)} chunks.")

    print("🧠 Generating embeddings...")
    embedded_chunks = embed_chunks(chunks)

    print(f"✅ Embeddings generated for {len(embedded_chunks)} chunks.")
    return embedded_chunks
