"""
BM25 Index Builder and Manager
-------------------------------
Builds and maintains BM25 indices for keyword-based retrieval.
Works alongside Milvus semantic search for hybrid retrieval.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import re
from rank_bm25 import BM25Okapi
from src.config.settings import settings


class BM25IndexManager:
    """
    Manages BM25 indices for all legal document collections.
    """
    
    def __init__(self, index_dir: Path = None):
        """
        Initialize BM25 index manager.
        
        Args:
            index_dir: Directory to store BM25 indices (default: data/bm25_indices)
        """
        self.index_dir = index_dir or settings.DATA_DIR / "bm25_indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.indices = {}  # collection_name -> BM25Okapi object
        self.documents = {}  # collection_name -> list of document dicts
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and normalize text for BM25 indexing.
        
        Legal-specific preprocessing:
        - Preserve section numbers (e.g., "Section 420")
        - Keep legal abbreviations (IPC, CrPC, etc.)
        - Remove stopwords but keep legal terms
        
        Args:
            text: Raw text content
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Preserve legal patterns
        text = re.sub(r'section\s+(\d+[a-z]*)', r'section_\1', text)
        text = re.sub(r'article\s+(\d+[a-z]*)', r'article_\1', text)
        
        # Tokenize (keep alphanumeric and underscores)
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens (but keep numbers for sections)
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
        
        # Common legal stopwords to KEEP (unlike general stopwords)
        legal_terms = {
            'act', 'section', 'ipc', 'crpc', 'cpc', 'law', 'court',
            'shall', 'liable', 'offense', 'punishment', 'rights'
        }
        
        # Remove common stopwords except legal ones
        stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are',
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'should', 'could', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'what', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such'
        }
        
        tokens = [t for t in tokens if t not in stopwords or t in legal_terms]
        
        return tokens
    
    def build_index_from_milvus(
        self,
        collection_name: str,
        client,
        batch_size: int = 1000
    ):
        """
        Build BM25 index from existing Milvus collection.
        
        Args:
            collection_name: Name of Milvus collection
            client: MilvusClient instance
            batch_size: Number of documents to process at once
        """
        print(f"\n[BM25] Building index for collection: {collection_name}")
        
        try:
            # Load collection
            from src.vector_store.milvus_store import load_collection
            load_collection(client, collection_name)
            
            # Get all documents (query with dummy vector)
            # In production, you'd iterate through all documents
            # For now, we'll use a workaround: query with large limit
            all_docs = []
            
            # Get collection stats to know document count
            stats = client.get_collection_stats(collection_name=collection_name)
            total_count = stats.get('row_count', 0)
            print(f"[BM25] Collection has {total_count} documents")
            
            # Retrieve documents in batches
            # Note: This is a simplified approach. In production, use iterator pattern
            dummy_vector = [0.0] * 768  # Adjust dimension based on your embeddings
            
            results = client.search(
                collection_name=collection_name,
                data=[dummy_vector],
                limit=min(total_count, 10000),  # Cap at 10k for memory
                output_fields=["content", "metadata"]
            )
            
            # Process results
            for hit in results[0]:
                entity = hit.get('entity', {})
                content = entity.get('content', '')
                metadata = entity.get('metadata', {})
                
                all_docs.append({
                    'content': content,
                    'metadata': metadata,
                    'id': hit.get('id')
                })
            
            print(f"[BM25] Retrieved {len(all_docs)} documents")
            
            # Tokenize all documents
            tokenized_corpus = [
                self.preprocess_text(doc['content'])
                for doc in all_docs
            ]
            
            # Build BM25 index
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Store index and documents
            self.indices[collection_name] = bm25
            self.documents[collection_name] = all_docs
            
            # Persist to disk
            self.save_index(collection_name)
            
            print(f"[BM25] ✅ Index built successfully for {collection_name}")
            print(f"[BM25]    - Documents: {len(all_docs)}")
            print(f"[BM25]    - Avg tokens/doc: {sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus):.1f}")
            
        except Exception as e:
            print(f"[BM25] ❌ Error building index: {e}")
            raise
    
    def build_index_from_json(
        self,
        collection_name: str,
        json_file: Path
    ):
        """
        Build BM25 index from chunked JSON files (alternative method).
        
        Args:
            collection_name: Name for the index
            json_file: Path to JSON file with chunks
        """
        print(f"\n[BM25] Building index from JSON: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            print(f"[BM25] Loaded {len(chunks)} chunks")
            
            # Tokenize corpus
            tokenized_corpus = [
                self.preprocess_text(chunk['content'])
                for chunk in chunks
            ]
            
            # Build BM25 index
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Store
            self.indices[collection_name] = bm25
            self.documents[collection_name] = chunks
            
            # Persist
            self.save_index(collection_name)
            
            print(f"[BM25] ✅ Index built from JSON for {collection_name}")
            
        except Exception as e:
            print(f"[BM25] ❌ Error building index from JSON: {e}")
            raise
    
    def save_index(self, collection_name: str):
        """
        Persist BM25 index to disk.
        """
        index_path = self.index_dir / f"{collection_name}_bm25.pkl"
        docs_path = self.index_dir / f"{collection_name}_docs.pkl"
        
        with open(index_path, 'wb') as f:
            pickle.dump(self.indices[collection_name], f)
        
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents[collection_name], f)
        
        print(f"[BM25] 💾 Saved index to {index_path}")
    
    def load_index(self, collection_name: str):
        """
        Load BM25 index from disk.
        """
        index_path = self.index_dir / f"{collection_name}_bm25.pkl"
        docs_path = self.index_dir / f"{collection_name}_docs.pkl"
        
        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(f"BM25 index not found for {collection_name}")
        
        with open(index_path, 'rb') as f:
            self.indices[collection_name] = pickle.load(f)
        
        with open(docs_path, 'rb') as f:
            self.documents[collection_name] = pickle.load(f)
        
        print(f"[BM25] 📂 Loaded index from {index_path}")
    
    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword matching.
        
        Args:
            collection_name: Name of collection to search
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of documents with BM25 scores
        """
        # Load index if not in memory
        if collection_name not in self.indices:
            self.load_index(collection_name)
        
        # Tokenize query
        tokenized_query = self.preprocess_text(query)
        
        # Get BM25 scores
        bm25 = self.indices[collection_name]
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            doc = self.documents[collection_name][idx]
            results.append({
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'score': float(scores[idx]),
                'rank': len(results) + 1,
                'retrieval_method': 'BM25'
            })
        
        return results
    
    def batch_build_all_collections(self, client):
        """
        Build BM25 indices for all Milvus collections.
        """
        collections = client.list_collections()
        print(f"\n[BM25] Building indices for {len(collections)} collections")
        
        for coll_name in collections:
            try:
                self.build_index_from_milvus(coll_name, client)
            except Exception as e:
                print(f"[BM25] ⚠️ Skipped {coll_name}: {e}")
                continue


# ===============================================================
# Build Indices Script
# ===============================================================
if __name__ == "__main__":
    from src.vector_store.milvus_store import connect_milvus
    
    print("="*80)
    print("BM25 INDEX BUILDER")
    print("="*80)
    
    # Connect to Milvus
    client = connect_milvus()
    
    # Initialize BM25 manager
    manager = BM25IndexManager()
    
    # Build indices for all collections
    manager.batch_build_all_collections(client)
    
    print("\n" + "="*80)
    print("✅ ALL BM25 INDICES BUILT SUCCESSFULLY")
    print("="*80)
    
    # Test search
    print("\n[TEST] Searching for 'Section 420 IPC punishment'")
    results = manager.search(
        collection_name="ipc_sections",
        query="Section 420 IPC punishment",
        top_k=3
    )
    
    for i, res in enumerate(results, 1):
        print(f"\n{i}. Score: {res['score']:.2f}")
        print(f"   {res['content'][:200]}...")