"""
Domain Routing Agent
--------------------
Classifies user queries into legal domains and routes to appropriate Milvus collections.

Supports:
- Multi-label classification (queries can span multiple domains)
- Confidence-based collection selection
- Fallback strategies for ambiguous queries
"""

import json
from typing import Dict, List, Tuple
from transformers import pipeline
import re


class DomainRoutingAgent:
    """
    Intelligent routing agent that maps legal queries to specific knowledge bases.
    """

    # Mapping of legal domains to Milvus collection names
    DOMAIN_TO_COLLECTION = {
        "Criminal Law": "criminal_law_chunks_store",
        "Civil Law": "civil_law_chunks_store",
        "Family Law": "family_law_chunks_store",
        "Property Law": "property_law_chunks_store",
        "Labour Law": "labour_chunks_store",
        "Business Law": "business_law_chunks_store",
        "Consumer Law": "consumer_law_chunks_store",
        "Constitutional Law": "constitutional_law_chunks_store",
        "Taxation": "taxation_chunks_store",
        "Intellectual Property": "intellectual_property_chunks_store",
        "IPC Sections": "ipc_sections"  # Fallback / general criminal code
    }

    # Keywords for rule-based enhancement (boosts classification confidence)
    DOMAIN_KEYWORDS = {
        "Criminal Law": [
            "fir", "police", "arrest", "bail", "murder", "theft", "assault",
            "robbery", "kidnapping", "criminal complaint", "chargesheet",
            "cognizable", "non-cognizable", "ipc", "crpc", "section 302",
            "section 420", "cheating", "fraud", "violence", "accused"
        ],
        "Civil Law": [
            "suit", "damages", "compensation", "civil court", "injunction",
            "declaration", "specific performance", "tort", "negligence",
            "breach of contract", "defamation", "nuisance"
        ],
        "Family Law": [
            "divorce", "marriage", "custody", "maintenance", "alimony",
            "adoption", "guardianship", "hindu marriage act", "muslim personal law",
            "domestic violence", "dowry", "child support", "separation",
            "matrimonial", "wife", "husband", "children"
        ],
        "Property Law": [
            "property", "land", "eviction", "tenant", "landlord", "lease",
            "rent", "title", "ownership", "possession", "transfer of property",
            "registry", "mutation", "encroachment", "easement", "mortgage",
            "sale deed", "partition", "inheritance"
        ],
        "Labour Law": [
            "employee", "employer", "termination", "resignation", "salary",
            "wages", "provident fund", "esi", "gratuity", "bonus",
            "industrial dispute", "trade union", "strike", "lockout",
            "minimum wages", "overtime", "workplace", "contract labour"
        ],
        "Business Law": [
            "company", "partnership", "shareholder", "director", "moa", "aoa",
            "incorporation", "winding up", "insolvency", "bankruptcy",
            "companies act", "nclt", "arbitration", "commercial dispute",
            "contract", "agreement", "business"
        ],
        "Consumer Law": [
            "consumer", "defective product", "service deficiency", "refund",
            "replacement", "consumer forum", "consumer protection act",
            "unfair trade practice", "warranty", "guarantee", "e-commerce"
        ],
        "Constitutional Law": [
            "fundamental rights", "article", "constitution", "writ", "pil",
            "habeas corpus", "mandamus", "supreme court", "high court",
            "judicial review", "directive principles", "amendment"
        ],
        "Taxation": [
            "income tax", "gst", "tax", "return", "assessment", "penalty",
            "itr", "tds", "tax evasion", "tax planning", "indirect tax",
            "custom duty", "excise"
        ],
        "Intellectual Property": [
            "patent", "trademark", "copyright", "design", "ip", "infringement",
            "licensing", "royalty", "piracy", "brand", "invention"
        ]
    }

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the routing agent with a zero-shot classifier.
        
        Args:
            model_name: HuggingFace model for classification
        """
        try:
            print(f"[INFO] Loading domain classifier: {model_name}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device_map='auto'
            )
            self.domains = list(self.DOMAIN_TO_COLLECTION.keys())
            print(f"[INFO] Domain routing agent initialized with {len(self.domains)} legal domains")
        except Exception as e:
            raise RuntimeError(f"Failed to load domain classifier: {e}")

    def classify_domain(
        self,
        query: str,
        threshold: float = 0.3,
        top_k: int = 3
    ) -> Dict[str, float]:
        """
        Classify query into one or more legal domains using zero-shot classification.
        
        Args:
            query: User's legal question
            threshold: Minimum confidence score to include a domain
            top_k: Maximum number of domains to return
            
        Returns:
            Dictionary mapping domain names to confidence scores
        """
        try:
            # Step 1: Zero-shot classification
            result = self.classifier(
                query,
                candidate_labels=self.domains,
                hypothesis_template="This legal query is related to {}.",
                multi_label=True  # Allow multiple domains
            )

            # Step 2: Keyword-based confidence boosting
            domain_scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                boosted_score = self._apply_keyword_boost(query, label, score)
                domain_scores[label] = boosted_score

            # Step 3: Filter by threshold and sort
            filtered_domains = {
                domain: score
                for domain, score in sorted(
                    domain_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                if score >= threshold
            }

            # Step 4: Fallback logic
            if not filtered_domains:
                print(f"[WARN] No domains above threshold. Using fallback: IPC Sections")
                filtered_domains = {"IPC Sections": 0.5}

            return filtered_domains

        except Exception as e:
            print(f"[ERROR] Domain classification failed: {e}")
            return {"IPC Sections": 0.5}  # Safe fallback

    def _apply_keyword_boost(
        self,
        query: str,
        domain: str,
        base_score: float,
        boost_factor: float = 0.15
    ) -> float:
        """
        Boost classification score if query contains domain-specific keywords.
        
        Args:
            query: User query (lowercased for matching)
            domain: Legal domain name
            base_score: Score from zero-shot classifier
            boost_factor: Amount to boost per keyword match
            
        Returns:
            Adjusted confidence score (capped at 1.0)
        """
        query_lower = query.lower()
        keywords = self.DOMAIN_KEYWORDS.get(domain, [])

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in query_lower)

        # Apply boost (diminishing returns)
        boost = min(matches * boost_factor, 0.3)  # Max boost of 30%
        adjusted_score = min(base_score + boost, 1.0)

        if matches > 0:
            print(f"[DEBUG] Domain '{domain}': {matches} keywords matched, "
                  f"score {base_score:.3f} → {adjusted_score:.3f}")

        return adjusted_score

    def route_to_collections(
        self,
        query: str,
        threshold: float = 0.3,
        max_collections: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Main routing function: maps query to Milvus collection names.
        
        Args:
            query: User's legal question
            threshold: Minimum confidence to route to a collection
            max_collections: Maximum collections to search
            
        Returns:
            List of (collection_name, confidence_score) tuples
        """
        # Classify domains
        domain_scores = self.classify_domain(query, threshold, max_collections)

        # Map to collection names
        collections = [
            (self.DOMAIN_TO_COLLECTION[domain], score)
            for domain, score in domain_scores.items()
            if domain in self.DOMAIN_TO_COLLECTION
        ]

        print(f"[INFO] Routed query to {len(collections)} collections:")
        for coll, score in collections:
            print(f"       - {coll} (confidence: {score:.3f})")

        return collections

    def get_routing_explanation(
        self,
        query: str,
        threshold: float = 0.3
    ) -> Dict:
        """
        Provide human-readable explanation of routing decision.
        
        Returns:
            Dictionary with routing details for transparency
        """
        domain_scores = self.classify_domain(query, threshold)
        collections = self.route_to_collections(query, threshold)

        return {
            "query": query,
            "identified_domains": domain_scores,
            "target_collections": [c[0] for c in collections],
            "routing_confidence": max([s for _, s in collections]) if collections else 0.0,
            "explanation": self._generate_explanation(domain_scores)
        }

    def _generate_explanation(self, domain_scores: Dict[str, float]) -> str:
        """
        Generate natural language explanation of routing decision.
        """
        if not domain_scores:
            return "Query is ambiguous. Routing to general legal knowledge base."

        primary_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[primary_domain]

        if confidence > 0.7:
            return f"High confidence: This appears to be a {primary_domain} query."
        elif confidence > 0.5:
            return f"Moderate confidence: Likely related to {primary_domain}."
        else:
            domains_str = ", ".join(domain_scores.keys())
            return f"Low confidence: Query may involve {domains_str}."


# ===============================================================
# Example Usage & Testing
# ===============================================================
if __name__ == "__main__":
    agent = DomainRoutingAgent()

    # Test queries spanning different domains
    test_queries = [
        "I sold my car to a person, who now is not interested in transferring it in his name.\r\n\r\nI received payment in cash.\r\nI have his ID (DL Photocopy)\r\nI have Delivery Note (just a paper form with basic information - not on stamp paper)\r\nI have photocopy of the RC Book\r\nThe RC-Book was handed over to him, and i signed some transfer related forms.\r\n\r\nEverytime I call him, he gives a new excuse.\r\nI have been following up for 2 months now without any success.\r\n\r\nI told him.that I will report the matter to police, he says he doesn't care. I have no idea how to proceed now. \r\n\r\nEven if I go to police or court, he will just repeat the lies like: \r\nhe will do it soon or he has no address proof or he wants to register in some other city or state. Full of clever excuses and expert at that.\r\n\r\nI even offered to do the transfer on his behalf - but he doesn't care.\r\nI have read that i can inform RTO etc - but practically i hear it does not help. Ultimately without a clear transfer - its no good. I need practical advice please.",
         "Sirs,\r\n\r\nIn absence of dissolution clause, if one/ two partner/s (out of 4 Partners) sends/ the notice of dissolution to other partners then my questions are - \r\n1. Can other partners continue the same business under the same firm name and style?\r\n2. Is that notice binding on other partners?\r\n3. If other partners wish to continue, can this notice be used to retire the partner/s who sent the notice of dissolution?\r\n4. If the notice is sent by two partners out of 4, can the remaining two continue?",
         "My wife wants to get divorced from me. We have been married for 3 years and have a 18 months old son. if we initiate divorce now by mutual consent how long will it take.till the time divorce proceedings are going on can my wife stop me from meeting my son.do i have to pay her maintenance till divorce is not granted.she is not working.i earn 1 lakh per month.what is the max maintenance she can demand.can she demand share on my property as well",
    ]

    print("\n" + "="*70)
    print("DOMAIN ROUTING AGENT - TEST RESULTS")
    print("="*70 + "\n")

    for query in test_queries:
        print(f"Query: {query}")
        explanation = agent.get_routing_explanation(query, threshold=0.25)
        print(json.dumps(explanation, indent=2))
        print("-" * 70 + "\n")