"""
Adaptive Hybrid Retriever
--------------------------
Automatically selects optimal fusion method and weights based on query analysis.

Features:
- Query-aware fusion selection
- Dynamic weight adjustment
- Fallback mechanisms
- Performance tracking
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import json
from pathlib import Path

from src.retriever.retriever import get_embedding
from src.vector_store.milvus_store import connect_milvus, load_collection, search_similar_chunks
from src.retriever.bm25_indexer import BM25IndexManager
from src.retriever.hybrid_fusion import HybridFusion
from src.retriever.query_analyzer import QueryAnalyzer, QueryCharacteristics
from src.config.settings import settings


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""
    query: str
    query_type: str
    fusion_method: str
    semantic_weight: float
    bm25_weight: float
    total_latency_ms: float
    semantic_latency_ms: float
    bm25_latency_ms: float
    fusion_latency_ms: float
    results_count: int
    collections_searched: List[str]


class AdaptiveHybridRetriever:
    """
    Intelligent retriever that adapts strategy based on query characteristics.
    """
    
    def __init__(
        self,
        enable_analytics: bool = True,
        analytics_file: Optional[Path] = None
    ):
        """
        Initialize adaptive retriever.
        
        Args:
            enable_analytics: Whether to track and log retrieval metrics
            analytics_file: Path to save analytics data
        """
        # Core components
        self.milvus_client = connect_milvus()
        self.bm25_manager = BM25IndexManager()
        self.fusion_engine = HybridFusion()
        self.query_analyzer = QueryAnalyzer()
        
        # Analytics
        self.enable_analytics = enable_analytics
        self.analytics_file = analytics_file or (settings.DATA_DIR / "retrieval_analytics.jsonl")
        self.session_metrics: List[RetrievalMetrics] = []
        
        print("[ADAPTIVE RETRIEVER] Initialized with query-aware fusion")
    
    def retrieve(
        self,
        query: str,
        collection_names: List[str],
        top_k: int = 5,
        force_fusion: Optional[str] = None,
        force_weights: Optional[tuple] = None
    ) -> tuple[List[Dict[str, Any]], QueryCharacteristics]:
        """
        Adaptive retrieval with automatic strategy selection.
        
        Args:
            query: User query
            collection_names: Milvus collections to search
            top_k: Number of final results
            force_fusion: Override fusion method (for testing)
            force_weights: Override weights (for testing)
            
        Returns:
            (results, query_characteristics)
        """
        start_time = time.time()
        
        # Step 1: Analyze query
        characteristics = self.query_analyzer.analyze(query)
        
        print(f"\n[ADAPTIVE] Query Type: {characteristics.query_type.value}")
        print(f"[ADAPTIVE] Recommended: {characteristics.recommended_fusion}")
        print(f"[ADAPTIVE] Weights: Sem={characteristics.recommended_weights[0]:.1f}, "
              f"BM25={characteristics.recommended_weights[1]:.1f}")
        
        # Step 2: Determine strategy (allow override for testing)
        fusion_method = force_fusion or characteristics.recommended_fusion
        weights = force_weights or characteristics.recommended_weights
        
        # Step 3: Retrieve from all collections
        all_results = []
        semantic_time = 0
        bm25_time = 0
        fusion_time = 0
        
        for collection_name in collection_names:
            print(f"\n[ADAPTIVE] Processing: {collection_name}")
            
            # Semantic search
            t0 = time.time()
            semantic_results = self._semantic_search(
                query, collection_name,
                top_k=15 if fusion_method != "learned" else 10
            )
            semantic_time += time.time() - t0
            
            # BM25 search
            t0 = time.time()
            bm25_results = self._bm25_search(
                query, collection_name,
                top_k=15 if fusion_method != "learned" else 10
            )
            bm25_time += time.time() - t0
            
            # Fusion
            t0 = time.time()
            fused_results = self._apply_fusion(
                query, semantic_results, bm25_results,
                method=fusion_method,
                weights=weights,
                top_k=top_k
            )
            fusion_time += time.time() - t0
            
            # Tag with source
            for result in fused_results:
                result['source_collection'] = collection_name
            
            all_results.extend(fused_results)
        
        # Step 4: Final re-ranking across collections
        all_results = self._cross_collection_rerank(
            all_results, fusion_method, top_k
        )
        
        total_time = time.time() - start_time
        
        # Step 5: Log analytics
        if self.enable_analytics:
            metrics = RetrievalMetrics(
                query=query,
                query_type=characteristics.query_type.value,
                fusion_method=fusion_method,
                semantic_weight=weights[0],
                bm25_weight=weights[1],
                total_latency_ms=total_time * 1000,
                semantic_latency_ms=semantic_time * 1000,
                bm25_latency_ms=bm25_time * 1000,
                fusion_latency_ms=fusion_time * 1000,
                results_count=len(all_results),
                collections_searched=collection_names
            )
            self._log_metrics(metrics)
        
        print(f"\n[ADAPTIVE] Retrieved {len(all_results)} results in {total_time*1000:.1f}ms")
        
        return all_results, characteristics
    
    def _semantic_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using Milvus."""
        try:
            embedding = get_embedding(query)
            load_collection(self.milvus_client, collection_name)
            results = search_similar_chunks(
                client=self.milvus_client,
                collection_name=collection_name,
                query_embedding=embedding,
                top_k=top_k
            )
            print(f"[SEMANTIC] {len(results)} results")
            return results
        except Exception as e:
            print(f"[SEMANTIC] Error: {e}")
            return []
    
    def _bm25_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        try:
            results = self.bm25_manager.search(
                collection_name=collection_name,
                query=query,
                top_k=top_k
            )
            print(f"[BM25] {len(results)} results")
            return results
        except Exception as e:
            print(f"[BM25] Error: {e}")
            return []
    
    def _apply_fusion(
        self,
        query: str,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        method: str,
        weights: tuple,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Apply selected fusion method with adaptive weights.
        """
        # Handle empty results
        if not semantic_results and not bm25_results:
            return []
        if not semantic_results:
            print("[FUSION] Only BM25 available")
            return bm25_results[:top_k]
        if not bm25_results:
            print("[FUSION] Only semantic available")
            return semantic_results[:top_k]
        
        # Apply fusion
        print(f"[FUSION] Method: {method}, Weights: {weights}")
        
        if method == "rrf":
            return self.fusion_engine.reciprocal_rank_fusion(
                semantic_results, bm25_results, top_k=top_k
            )
        
        elif method == "weighted":
            return self.fusion_engine.weighted_fusion(
                semantic_results, bm25_results,
                semantic_weight=weights[0],
                bm25_weight=weights[1],
                top_k=top_k
            )
        
        elif method == "learned":
            return self.fusion_engine.learned_fusion(
                query, semantic_results, bm25_results, top_k=top_k
            )
        
        else:
            # Fallback to RRF
            print(f"[FUSION] Unknown method '{method}', falling back to RRF")
            return self.fusion_engine.reciprocal_rank_fusion(
                semantic_results, bm25_results, top_k=top_k
            )
    
    def _cross_collection_rerank(
        self,
        results: List[Dict[str, Any]],
        fusion_method: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results across all collections.
        """
        # Get score key based on fusion method
        score_keys = {
            "rrf": "rrf_score",
            "weighted": "weighted_score",
            "learned": "rerank_score"
        }
        score_key = score_keys.get(fusion_method, "score")
        
        # Sort by score
        results.sort(
            key=lambda x: x.get(score_key, 0),
            reverse=True
        )
        
        return results[:top_k]
    
    def _log_metrics(self, metrics: RetrievalMetrics):
        """
        Log retrieval metrics to file and session.
        """
        self.session_metrics.append(metrics)
        
        # Append to JSONL file
        try:
            with open(self.analytics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(metrics)) + '\n')
        except Exception as e:
            print(f"[ANALYTICS] Failed to log: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for current session.
        """
        if not self.session_metrics:
            return {"message": "No retrievals in this session"}
        
        # Calculate averages by query type
        by_type = {}
        for m in self.session_metrics:
            qtype = m.query_type
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(m)
        
        summary = {
            "total_queries": len(self.session_metrics),
            "average_latency_ms": sum(m.total_latency_ms for m in self.session_metrics) / len(self.session_metrics),
            "by_query_type": {}
        }
        
        for qtype, metrics in by_type.items():
            summary["by_query_type"][qtype] = {
                "count": len(metrics),
                "avg_latency_ms": sum(m.total_latency_ms for m in metrics) / len(metrics),
                "most_used_fusion": max(
                    set(m.fusion_method for m in metrics),
                    key=lambda x: sum(1 for m in metrics if m.fusion_method == x)
                )
            }
        
        return summary
    
    def print_session_summary(self):
        """
        Print formatted session summary.
        """
        summary = self.get_session_summary()
        
        print("\n" + "="*80)
        print("RETRIEVAL SESSION SUMMARY")
        print("="*80)
        print(f"\nTotal Queries: {summary['total_queries']}")
        print(f"Average Latency: {summary['average_latency_ms']:.1f}ms")
        
        print("\nBy Query Type:")
        print("-"*80)
        for qtype, stats in summary.get('by_query_type', {}).items():
            print(f"\n{qtype.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.1f}ms")
            print(f"  Most Used Fusion: {stats['most_used_fusion']}")
        
        print("\n" + "="*80)


# ===============================================================
# Drop-in Replacement for RAG Graph
# ===============================================================
def adaptive_retrieve_node(state):
    """
    Enhanced retrieval node with query-adaptive fusion.
    Compatible with existing RAGState structure.
    """
    try:
        # Initialize adaptive retriever
        retriever = AdaptiveHybridRetriever(enable_analytics=True)
        
        # Adaptive retrieval
        results, characteristics = retriever.retrieve(
            query=state["user_query"],
            collection_names=state["target_collections"],
            top_k=5
        )
        
        # Update state with results and analysis
        state["retrieved_chunks"] = results
        state["query_characteristics"] = characteristics  # New field
        
        # Add retrieval explanation to state
        state["retrieval_explanation"] = (
            f"Used {characteristics.recommended_fusion} fusion "
            f"(optimized for {characteristics.query_type.value} queries)"
        )
        
        print(f"[ADAPTIVE] Completed: {len(results)} chunks retrieved")
        return state
        
    except Exception as e:
        print(f"[ADAPTIVE] Error: {e}")
        state["retrieved_chunks"] = []
        return state


# ===============================================================
# Standalone Testing
# ===============================================================
if __name__ == "__main__":
    print("="*80)
    print("ADAPTIVE HYBRID RETRIEVER - TEST SUITE")
    print("="*80)
    
    # Initialize
    retriever = AdaptiveHybridRetriever(enable_analytics=True)
    
    # Test queries of different types
    test_cases = [
        # (query, expected_collections)
        ("Section 420 IPC punishment", ["ipc_sections"]),
        ("What is bail and how does it work?", ["criminal_law_chunks_store"]),
        ("How to file for divorce under Hindu Marriage Act?", ["family_law_chunks_store"]),
        ("My landlord evicted me without notice", ["property_law_chunks_store"]),
        ("Landlord filed false FIR and property dispute", ["property_law_chunks_store", "criminal_law_chunks_store"])
    ]
    
    for query, collections in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {query}")
        print(f"{'='*80}")
        
        results, characteristics = retriever.retrieve(
            query=query,
            collection_names=collections,
            top_k=3
        )
        
        print(f"\nResults ({len(results)}):")
        for i, result in enumerate(results, 1):
            score_key = {
                "rrf": "rrf_score",
                "weighted": "weighted_score",
                "learned": "rerank_score"
            }.get(characteristics.recommended_fusion, "score")
            
            print(f"\n{i}. [Score: {result.get(score_key, 0):.4f}]")
            print(f"   {result['content'][:150]}...")
            print(f"   Collection: {result.get('source_collection', 'unknown')}")
    
    # Print session summary
    retriever.print_session_summary()