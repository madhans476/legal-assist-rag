"""
Query Analyzer & Classifier
----------------------------
Analyzes query characteristics to determine optimal retrieval strategy.

Query Types:
1. EXACT_MATCH: "Section 420 IPC", "Article 21"
2. CONCEPTUAL: "What are my rights?", "Explain bail"
3. PROCEDURAL: "How to file FIR?", "Divorce process"
4. CASE_BASED: "Can I sue my landlord?", specific situations
5. MULTI_ASPECT: "Landlord eviction and false FIR" (multiple domains)
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Enum for different query types."""
    EXACT_MATCH = "exact_match"       # Section/article lookups
    CONCEPTUAL = "conceptual"         # Definition/explanation requests
    PROCEDURAL = "procedural"         # How-to/process queries
    CASE_BASED = "case_based"         # Specific situation analysis
    MULTI_ASPECT = "multi_aspect"     # Complex multi-domain queries


@dataclass
class QueryCharacteristics:
    """Data class holding query analysis results."""
    query_type: QueryType
    confidence: float
    has_legal_references: bool
    has_section_numbers: bool
    has_question_words: bool
    complexity_score: float
    entity_count: int
    recommended_fusion: str
    recommended_weights: Tuple[float, float]  # (semantic, bm25)
    reasoning: str


class QueryAnalyzer:
    """
    Analyzes queries to determine optimal retrieval configuration.
    """
    
    # Legal reference patterns
    SECTION_PATTERN = re.compile(
        r'\b(section|sec|s\.?)\s*(\d+[a-z]*)\b',
        re.IGNORECASE
    )
    
    ARTICLE_PATTERN = re.compile(
        r'\b(article|art\.?)\s*(\d+[a-z]*)\b',
        re.IGNORECASE
    )
    
    ACT_PATTERN = re.compile(
        r'\b(ipc|crpc|cpc|hindu marriage act|evidence act|constitution)\b',
        re.IGNORECASE
    )
    
    # Question patterns
    QUESTION_WORDS = [
        'what', 'how', 'why', 'when', 'where', 'which', 'who',
        'can', 'should', 'could', 'would', 'is', 'are', 'does'
    ]
    
    # Procedural indicators
    PROCEDURAL_KEYWORDS = [
        'process', 'procedure', 'steps', 'how to', 'file', 'register',
        'apply', 'obtain', 'get', 'claim', 'lodge', 'submit'
    ]
    
    # Conceptual indicators
    CONCEPTUAL_KEYWORDS = [
        'definition', 'define', 'meaning', 'what is', 'explain',
        'describe', 'understand', 'concept', 'difference between'
    ]
    
    # Case-based indicators
    CASE_INDICATORS = [
        'my', 'I', 'me', 'situation', 'case', 'happened', 'can i',
        'should i', 'landlord', 'employer', 'spouse', 'tenant'
    ]
    
    def __init__(self):
        """Initialize query analyzer."""
        print("[QUERY ANALYZER] Initialized")
    
    def analyze(self, query: str) -> QueryCharacteristics:
        """
        Comprehensive query analysis.
        
        Args:
            query: User query string
            
        Returns:
            QueryCharacteristics object with analysis results
        """
        query_lower = query.lower()
        
        # Extract features
        has_sections = bool(self.SECTION_PATTERN.search(query))
        has_articles = bool(self.ARTICLE_PATTERN.search(query))
        has_acts = bool(self.ACT_PATTERN.search(query))
        has_legal_refs = has_sections or has_articles or has_acts
        
        # Question analysis
        has_questions = any(
            qw in query_lower for qw in self.QUESTION_WORDS
        )
        
        # Keyword matching
        procedural_count = sum(
            1 for kw in self.PROCEDURAL_KEYWORDS if kw in query_lower
        )
        conceptual_count = sum(
            1 for kw in self.CONCEPTUAL_KEYWORDS if kw in query_lower
        )
        case_count = sum(
            1 for kw in self.CASE_INDICATORS if kw in query_lower
        )
        
        # Entity counting (rough proxy for complexity)
        entities = self._count_entities(query)
        
        # Complexity score (0-1)
        complexity = self._calculate_complexity(
            query, entities, has_legal_refs
        )
        
        # Classify query type
        query_type, confidence = self._classify_query_type(
            has_sections, has_articles, procedural_count,
            conceptual_count, case_count, complexity
        )
        
        # Determine optimal fusion strategy
        fusion_method, weights, reasoning = self._recommend_fusion(
            query_type, has_legal_refs, complexity
        )
        
        return QueryCharacteristics(
            query_type=query_type,
            confidence=confidence,
            has_legal_references=has_legal_refs,
            has_section_numbers=(has_sections or has_articles),
            has_question_words=has_questions,
            complexity_score=complexity,
            entity_count=entities,
            recommended_fusion=fusion_method,
            recommended_weights=weights,
            reasoning=reasoning
        )
    
    def _count_entities(self, query: str) -> int:
        """
        Count legal entities (sections, articles, acts, proper nouns).
        """
        entity_count = 0
        
        # Section/Article mentions
        entity_count += len(self.SECTION_PATTERN.findall(query))
        entity_count += len(self.ARTICLE_PATTERN.findall(query))
        entity_count += len(self.ACT_PATTERN.findall(query))
        
        # Proper nouns (capitalized words, rough heuristic)
        words = query.split()
        entity_count += sum(
            1 for w in words if w[0].isupper() and len(w) > 2
        )
        
        return entity_count
    
    def _calculate_complexity(
        self,
        query: str,
        entity_count: int,
        has_legal_refs: bool
    ) -> float:
        """
        Calculate query complexity score (0-1).
        
        Factors:
        - Length (longer = more complex)
        - Number of entities
        - Presence of conjunctions (and, or)
        - Legal references
        """
        # Length factor (0-1, sigmoid)
        words = query.split()
        length_score = min(len(words) / 30, 1.0)
        
        # Entity factor
        entity_score = min(entity_count / 5, 1.0)
        
        # Conjunction factor (indicates multi-aspect)
        conjunctions = sum(
            1 for conj in [' and ', ' or ', ','] if conj in query.lower()
        )
        conjunction_score = min(conjunctions / 3, 1.0)
        
        # Legal reference factor
        legal_score = 0.3 if has_legal_refs else 0.0
        
        # Weighted combination
        complexity = (
            0.3 * length_score +
            0.3 * entity_score +
            0.2 * conjunction_score +
            0.2 * legal_score
        )
        
        return min(complexity, 1.0)
    
    def _classify_query_type(
        self,
        has_sections: bool,
        has_articles: bool,
        procedural_count: int,
        conceptual_count: int,
        case_count: int,
        complexity: float
    ) -> Tuple[QueryType, float]:
        """
        Classify query into one of five types with confidence.
        
        Returns:
            (QueryType, confidence_score)
        """
        scores = {
            QueryType.EXACT_MATCH: 0.0,
            QueryType.CONCEPTUAL: 0.0,
            QueryType.PROCEDURAL: 0.0,
            QueryType.CASE_BASED: 0.0,
            QueryType.MULTI_ASPECT: 0.0
        }
        
        # Exact match scoring
        if has_sections or has_articles:
            scores[QueryType.EXACT_MATCH] += 0.8
        
        # Conceptual scoring
        if conceptual_count > 0:
            scores[QueryType.CONCEPTUAL] += 0.3 * conceptual_count
        
        # Procedural scoring
        if procedural_count > 0:
            scores[QueryType.PROCEDURAL] += 0.3 * procedural_count
        
        # Case-based scoring
        if case_count > 0:
            scores[QueryType.CASE_BASED] += 0.25 * case_count
        
        # Multi-aspect scoring (high complexity)
        if complexity > 0.6:
            scores[QueryType.MULTI_ASPECT] += complexity
        
        # Get top type
        top_type = max(scores, key=scores.get)
        confidence = min(scores[top_type], 1.0)
        
        # Fallback to conceptual if all scores low
        if confidence < 0.2:
            return QueryType.CONCEPTUAL, 0.5
        
        return top_type, confidence
    
    def _recommend_fusion(
        self,
        query_type: QueryType,
        has_legal_refs: bool,
        complexity: float
    ) -> Tuple[str, Tuple[float, float], str]:
        """
        Recommend fusion method and weights based on query analysis.
        
        Returns:
            (fusion_method, (semantic_weight, bm25_weight), reasoning)
        """
        if query_type == QueryType.EXACT_MATCH:
            # Exact matches: BM25 excels at keyword matching
            return (
                "weighted",
                (0.3, 0.7),  # Favor BM25
                "Exact legal reference detected. BM25 weighted higher for precise section matching."
            )
        
        elif query_type == QueryType.CONCEPTUAL:
            # Conceptual: Semantic search better for understanding
            return (
                "weighted",
                (0.7, 0.3),  # Favor semantic
                "Conceptual query. Semantic search weighted higher for meaning-based retrieval."
            )
        
        elif query_type == QueryType.PROCEDURAL:
            # Procedural: Balanced, slight semantic preference
            return (
                "rrf",
                (0.6, 0.4),
                "Procedural query. RRF fusion for balanced keyword and semantic matching."
            )
        
        elif query_type == QueryType.CASE_BASED:
            # Case-based: Needs deep understanding, use learned
            if complexity > 0.5:
                return (
                    "learned",
                    (0.5, 0.5),
                    "Complex case-based query. Cross-encoder re-ranking for accurate situation matching."
                )
            else:
                return (
                    "weighted",
                    (0.6, 0.4),
                    "Simple case-based query. Semantic-weighted fusion for context understanding."
                )
        
        elif query_type == QueryType.MULTI_ASPECT:
            # Multi-aspect: Use learned fusion for best results
            return (
                "learned",
                (0.5, 0.5),
                "Multi-aspect query spanning multiple legal domains. Cross-encoder for comprehensive matching."
            )
        
        else:
            # Default fallback
            return (
                "rrf",
                (0.5, 0.5),
                "Standard query. RRF fusion for robust baseline performance."
            )
    
    def explain_analysis(self, characteristics: QueryCharacteristics) -> str:
        """
        Generate human-readable explanation of analysis.
        """
        explanation = f"""
Query Analysis Report
{'='*60}

Query Type: {characteristics.query_type.value.upper()}
Confidence: {characteristics.confidence:.2%}

Detected Features:
  - Legal References: {'Yes' if characteristics.has_legal_references else 'No'}
  - Section/Article Numbers: {'Yes' if characteristics.has_section_numbers else 'No'}
  - Question Words: {'Yes' if characteristics.has_question_words else 'No'}
  - Entity Count: {characteristics.entity_count}
  - Complexity Score: {characteristics.complexity_score:.2f}/1.00

Recommended Strategy:
  - Fusion Method: {characteristics.recommended_fusion.upper()}
  - Weights: Semantic={characteristics.recommended_weights[0]:.1f}, BM25={characteristics.recommended_weights[1]:.1f}

Reasoning:
  {characteristics.reasoning}

{'='*60}
"""
        return explanation


# ===============================================================
# Testing & Examples
# ===============================================================
if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    
    # Test queries covering all types
    test_queries = [
        # EXACT_MATCH
        "Section 420 IPC punishment",
        "Article 21 of Constitution",
        
        # CONCEPTUAL
        "What is bail?",
        "Explain the concept of anticipatory bail",
        "Definition of cheating under IPC",
        
        # PROCEDURAL
        "How to file an FIR?",
        "Steps to register a company",
        "Divorce procedure under Hindu Marriage Act",
        
        # CASE_BASED
        "My landlord evicted me without notice",
        "Can I sue my employer for wrongful termination?",
        "I was falsely accused of theft, what should I do?",
        
        # MULTI_ASPECT
        "Landlord filed false FIR and won't return deposit",
        "Divorce process and child custody rights in India",
        "Property dispute with criminal intimidation charges"
    ]
    
    print("\n" + "="*80)
    print("QUERY ANALYZER - TEST SUITE")
    print("="*80 + "\n")
    
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 80)
        
        characteristics = analyzer.analyze(query)
        
        print(f"Type: {characteristics.query_type.value}")
        print(f"Confidence: {characteristics.confidence:.2%}")
        print(f"Complexity: {characteristics.complexity_score:.2f}")
        print(f"Recommended: {characteristics.recommended_fusion} "
              f"(Sem:{characteristics.recommended_weights[0]:.1f}, "
              f"BM25:{characteristics.recommended_weights[1]:.1f})")
        print(f"Reason: {characteristics.reasoning}")
        
        # Detailed explanation for first query
        if query == test_queries[0]:
            print("\n" + analyzer.explain_analysis(characteristics))