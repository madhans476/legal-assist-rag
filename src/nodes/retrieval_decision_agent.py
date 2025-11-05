import json
from typing import Dict
from transformers import pipeline


class RetrievalDecisionAgent:
    """
    Improved Retrieval Decision Agent.
    Determines whether a user query requires retrieval from legal knowledge base.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        try:
            # Smaller and faster zero-shot classification model
            self.classifier = pipeline("zero-shot-classification", model=model_name)
            print(f"[INFO] RetrievalDecisionAgent initialized with {model_name}")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def analyze_query(self, query: str) -> Dict[str, bool]:
        """
        Decide if retrieval from external legal knowledge is needed.
        """
        try:
            candidate_labels = ["retrieval_needed", "no_retrieval_needed"]
    
            # Include {} in the hypothesis template
            hypothesis_template = "This query {} retrieval from legal documents to answer."
            # hypothesis_template = "This query {} from Indian legal statutes, documents, acts, or case law to answer correctly."
    
            result = self.classifier(query, candidate_labels, hypothesis_template=hypothesis_template)
            top_label = result["labels"][0]
            confidence = result["scores"][0]
    
            retrieval_needed = top_label == "retrieval_needed"
    
            return {
                "retrieval_needed": retrieval_needed,
                "confidence": round(confidence, 3),
                "reason": "zero-shot classification"
            }
    
        except Exception as e:
            print(f"[ERROR] Query analysis failed: {e}")
            return {"retrieval_needed": True, "reason": "error during decision"}



# Example usage
if __name__ == "__main__":
    agent = RetrievalDecisionAgent()
    examples = [
    # Retrieval needed
    "Explain Section 300 of IPC.",
    "What are the rights under Article 21 of the Indian Constitution?",
    "List the rights of consumers under the Consumer Protection Act, 2019.",
    "Explain the grounds for divorce under Hindu Marriage Act.",
    "What is the punishment for theft under IPC?",
    "Can I be jailed for not paying taxes?",
    "When does self-defense become murder?",
    "How can I register a trademark?",

    # No retrieval needed
    "Hi, how are you?",
    "Who are you?",
    "What is artificial intelligence?",
    "Explain photosynthesis.",
    "What’s the weather like in Delhi?",
    "Give me an inspirational quote about law.",
    "Who was Dr. B.R. Ambedkar?",
    "Define justice in general terms.",
    "What is crime?",
    "Difference between law and ethics.",
    "What is newton's law?",
    "What are the law related to overtime in working hours?"
]
    for q in examples:
        decision = agent.analyze_query(q)
        print(f"\nQuery: {q}")
        print(json.dumps(decision, indent=2))
