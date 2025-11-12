"""
Hybrid Retrieval Fusion Methods
--------------------------------
Combines semantic search (Milvus) and keyword search (BM25) results.

Implements three fusion strategies:
1. Reciprocal Rank Fusion (RRF) - Standard approach
2. Weighted Fusion - Configurable weight for each method
3. Learned Fusion - Uses cross-encoder for re-ranking
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
from sentence_transformers import CrossEncoder


class HybridFusion:
    """
    Combines results from multiple retrieval methods.
    """
    
    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize fusion engine with optional re-ranker.
        
        Args:
            reranker_model: HuggingFace cross-encoder model for learned fusion
        """
        self.reranker = None
        self.reranker_model_name = reranker_model
        
    def _load_reranker(self):
        """Lazy load cross-encoder model (only when needed)."""
        if self.reranker is None:
            print(f"[FUSION] Loading cross-encoder: {self.reranker_model_name}")
            self.reranker = CrossEncoder(self.reranker_model_name)
            print(f"[FUSION] ✅ Cross-encoder loaded")
    
    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        
        Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
        
        Args:
            semantic_results: Results from Milvus semantic search
            bm25_results: Results from BM25 keyword search
            k: Constant to avoid division by zero (default: 60)
            top_k: Number of final results
            
        Returns:
            Fused and re-ranked results
        """
        # Build document index with RRF scores
        doc_scores = defaultdict(lambda: {
            'rrf_score': 0.0,
            'semantic_rank': None,
            'bm25_rank': None,
            'semantic_score': None,
            'bm25_score': None,
            'content': None,
            'metadata': None
        })
        
        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = self._get_doc_id(result)
            doc_scores[doc_id]['rrf_score'] += 1 / (k + rank)
            doc_scores[doc_id]['semantic_rank'] = rank
            doc_scores[doc_id]['semantic_score'] = result.get('score', 0)
            doc_scores[doc_id]['content'] = result['content']
            doc_scores[doc_id]['metadata'] = result.get('metadata', {})
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = self._get_doc_id(result)
            doc_scores[doc_id]['rrf_score'] += 1 / (k + rank)
            doc_scores[doc_id]['bm25_rank'] = rank
            doc_scores[doc_id]['bm25_score'] = result.get('score', 0)
            
            # Update content/metadata if not set
            if doc_scores[doc_id]['content'] is None:
                doc_scores[doc_id]['content'] = result['content']
                doc_scores[doc_id]['metadata'] = result.get('metadata', {})
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]['rrf_score'],
            reverse=True
        )[:top_k]
        
        # Format results
        fused_results = []
        for doc_id, scores in sorted_docs:
            fused_results.append({
                'content': scores['content'],
                'metadata': scores['metadata'],
                'rrf_score': scores['rrf_score'],
                'semantic_rank': scores['semantic_rank'],
                'bm25_rank': scores['bm25_rank'],
                'semantic_score': scores['semantic_score'],
                'bm25_score': scores['bm25_score'],
                'fusion_method': 'RRF'
            })
        
        return fused_results
    
    def weighted_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Weighted linear combination of normalized scores.
        
        Formula: final_score = α * norm_semantic + β * norm_bm25
        
        Args:
            semantic_results: Results from Milvus
            bm25_results: Results from BM25
            semantic_weight: Weight for semantic scores (α)
            bm25_weight: Weight for BM25 scores (β)
            top_k: Number of results
            
        Returns:
            Weighted fused results
        """
        # Normalize scores to [0, 1]
        semantic_scores_norm = self._normalize_scores(
            [r.get('score', 0) for r in semantic_results]
        )
        bm25_scores_norm = self._normalize_scores(
            [r.get('score', 0) for r in bm25_results]
        )
        
        # Build document index
        doc_scores = defaultdict(lambda: {
            'weighted_score': 0.0,
            'content': None,
            'metadata': None,
            'semantic_score': None,
            'bm25_score': None
        })
        
        # Add semantic scores
        for result, norm_score in zip(semantic_results, semantic_scores_norm):
            doc_id = self._get_doc_id(result)
            doc_scores[doc_id]['weighted_score'] += semantic_weight * norm_score
            doc_scores[doc_id]['semantic_score'] = result.get('score', 0)
            doc_scores[doc_id]['content'] = result['content']
            doc_scores[doc_id]['metadata'] = result.get('metadata', {})
        
        # Add BM25 scores
        for result, norm_score in zip(bm25_results, bm25_scores_norm):
            doc_id = self._get_doc_id(result)
            doc_scores[doc_id]['weighted_score'] += bm25_weight * norm_score
            doc_scores[doc_id]['bm25_score'] = result.get('score', 0)
            
            if doc_scores[doc_id]['content'] is None:
                doc_scores[doc_id]['content'] = result['content']
                doc_scores[doc_id]['metadata'] = result.get('metadata', {})
        
        # Sort and return
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )[:top_k]
        
        fused_results = []
        for doc_id, scores in sorted_docs:
            fused_results.append({
                'content': scores['content'],
                'metadata': scores['metadata'],
                'weighted_score': scores['weighted_score'],
                'semantic_score': scores['semantic_score'],
                'bm25_score': scores['bm25_score'],
                'semantic_weight': semantic_weight,
                'bm25_weight': bm25_weight,
                'fusion_method': 'Weighted'
            })
        
        return fused_results
    
    def learned_fusion(
        self,
        query: str,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10,
        rerank_top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Use cross-encoder to re-rank combined results.
        
        Steps:
        1. Merge semantic + BM25 results
        2. Take top-N candidates
        3. Re-rank with cross-encoder (query, document) pairs
        4. Return top-K
        
        Args:
            query: User query
            semantic_results: Milvus results
            bm25_results: BM25 results
            top_k: Final number of results
            rerank_top_n: Candidates to re-rank (more = slower but better)
            
        Returns:
            Re-ranked results
        """
        # Load cross-encoder
        self._load_reranker()
        
        # Merge results (deduplicate)
        seen_ids = set()
        merged_results = []
        
        for result in semantic_results + bm25_results:
            doc_id = self._get_doc_id(result)
            if doc_id not in seen_ids:
                merged_results.append(result)
                seen_ids.add(doc_id)
        
        # Take top-N candidates for re-ranking
        candidates = merged_results[:rerank_top_n]
        
        # Prepare query-document pairs
        pairs = [[query, doc['content']] for doc in candidates]
        
        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Attach scores to documents
        for doc, score in zip(candidates, rerank_scores):
            doc['rerank_score'] = float(score)
        
        # Sort by re-rank score
        reranked = sorted(
            candidates,
            key=lambda x: x['rerank_score'],
            reverse=True
        )[:top_k]
        
        # Format results
        for doc in reranked:
            doc['fusion_method'] = 'Learned (Cross-Encoder)'
        
        return reranked
    
    def _get_doc_id(self, result: Dict[str, Any]) -> str:
        """
        Generate unique document ID for deduplication.
        Uses content hash or metadata identifier.
        """
        # Try to use metadata section/title as ID
        metadata = result.get('metadata', {})
        if 'section_no' in metadata:
            return f"section_{metadata['section_no']}"
        elif 'parent_id' in metadata:
            return f"parent_{metadata['parent_id']}"
        else:
            # Fallback: hash content
            return str(hash(result['content'][:100]))
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-max normalization to [0, 1].
        """
        if not scores:
            return []
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return ((scores - min_score) / (max_score - min_score)).tolist()
    
    def compare_fusion_methods(
        self,
        query: str,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all fusion methods and compare results.
        
        Returns:
            Dictionary with results from each method
        """
        results = {
            'rrf': self.reciprocal_rank_fusion(
                semantic_results, bm25_results, top_k=top_k
            ),
            'weighted': self.weighted_fusion(
                semantic_results, bm25_results, top_k=top_k
            ),
            'learned': self.learned_fusion(
                query, semantic_results, bm25_results, top_k=top_k
            )
        }
        
        return results


# ===============================================================
# Testing & Comparison
# ===============================================================
if __name__ == "__main__":
    # Mock results for testing
    semantic_results = [
        {
            'content': 'Section 420: Whoever cheats and thereby dishonestly induces...',
            'metadata': {'section_no': '420'},
            'score': 0.89
        },
        {
            'content': 'Cheating is defined as intentional deception...',
            'metadata': {'title': 'Fraud definition'},
            'score': 0.82
        },
        {
            'content': 'IPC deals with criminal offenses in India...',
            'metadata': {'title': 'IPC overview'},
            'score': 0.75
        }
    ]
    
    bm25_results = [
        {
            'content': 'Section 420: Whoever cheats and thereby dishonestly induces...',
            'metadata': {'section_no': '420'},
            'score': 18.5
        },
        {
            'content': 'Section 415: Cheating is defined as...',
            'metadata': {'section_no': '415'},
            'score': 12.3
        },
        {
            'content': 'Punishment under IPC for fraud...',
            'metadata': {'title': 'Punishment guide'},
            'score': 8.7
        }
    ]
    
    fusion = HybridFusion()
    
    print("="*80)
    print("FUSION METHOD COMPARISON")
    print("="*80)
    
    query = "Section 420 IPC cheating punishment"
    
    results = fusion.compare_fusion_methods(
        query, semantic_results, bm25_results, top_k=3
    )
    
    for method, docs in results.items():
        print(f"\n{method.upper()} RESULTS:")
        print("-"*80)
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc['content'][:80]}...")
            if 'rrf_score' in doc:
                print(f"   RRF: {doc['rrf_score']:.4f} | Sem rank: {doc['semantic_rank']} | BM25 rank: {doc['bm25_rank']}")
            elif 'weighted_score' in doc:
                print(f"   Weighted: {doc['weighted_score']:.4f}")
            elif 'rerank_score' in doc:
                print(f"   Rerank: {doc['rerank_score']:.4f}")