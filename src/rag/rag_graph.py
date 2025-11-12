# from typing import TypedDict, List, Dict
# from langgraph.graph import StateGraph, START, END
# from src.nodes.retrieval_decision_agent import RetrievalDecisionAgent
# from src.nodes.domain_routing_agent import DomainRoutingAgent
# from src.retriever.retriever import get_embedding
# from src.vector_store.milvus_store import connect_milvus, load_collection, search_similar_chunks
# from src.llm.llm_client import load_llm
# from src.config.settings import settings

# class RAGState(TypedDict):
#     user_query: str
#     query_embedding: List[float]
#     retrieval_needed: bool
#     target_domains: Dict[str, float]
#     target_collections: List[str]
#     retrieved_chunks: List[dict]
#     response: str
#     citations: List[str]
#     routing_explanation: str 

# # --- Node 1: Retrieval Decision ---
# def retrieval_decision_node(state: RAGState) -> RAGState:
#     try:
#         agent = RetrievalDecisionAgent()
#         decision = agent.analyze_query(state["user_query"])
#         state["retrieval_needed"] = decision["retrieval_needed"]
#         print(f"[INFO] Decision: {decision}")
#         return state
#     except Exception as e:
#         print(f"[ERROR] Retrieval decision failed: {e}")
#         state["retrieval_needed"] = True  # Fallback to safe path
#         return state

# # --- Node 3: Domain Routing ---
# def domain_routing_node(state: RAGState) -> RAGState:
#     """
#     Classifies query into legal domains and routes to appropriate collections.
#     """
#     try:
#         router = DomainRoutingAgent()
        
#         # Get domain classifications with confidence scores
#         domain_scores = router.classify_domain(
#             query=state["user_query"],
#             threshold=0.25,  # Lower threshold for multi-domain queries
#             top_k=3
#         )
        
#         # Map domains to collection names
#         collections_with_scores = router.route_to_collections(
#             query=state["user_query"],
#             threshold=0.25,
#             max_collections=3
#         )
        
#         # Extract collection names (sorted by confidence)
#         target_collections = [coll for coll, _ in collections_with_scores]
        
#         # Get human-readable explanation
#         explanation = router.get_routing_explanation(state["user_query"])
        
#         # Update state
#         state["target_domains"] = domain_scores
#         state["target_collections"] = target_collections
#         state["routing_explanation"] = explanation["explanation"]
        
#         print(f"[INFO] Domain Routing Complete:")
#         print(f"       Domains: {list(domain_scores.keys())}")
#         print(f"       Collections: {target_collections}")
        
#         return state
    
#     except Exception as e:
#         print(f"[ERROR] Domain routing failed: {e}")
#         # Fallback to IPC sections
#         state["target_domains"] = {"IPC Sections": 0.5}
#         state["target_collections"] = ["ipc_sections"]
#         state["routing_explanation"] = "Routing to general legal database due to classification error"
#         return state
    
# # --- Node 3: Retriever ---
# def retrieve_node(state: RAGState) -> RAGState:
#     """
#     Retrieves relevant chunks from multiple Milvus collections in parallel.
#     """
#     try:
#         # Generate query embedding once
#         embedding = get_embedding(state["user_query"])
#         state["query_embedding"] = embedding

#         # Connect to Milvus
#         client = connect_milvus()

#         all_chunks = []
#         top_k_per_collection = 3  # Retrieve top 3 from each collection

#         # Retrieve from each target collection
#         for collection_name in state["target_collections"]:
#             try:
#                 # Load collection into memory
#                 load_collection(client=client, collection_name=collection_name)
                
#                 # Search for similar chunks
#                 results = search_similar_chunks(
#                     client=client,
#                     collection_name=collection_name,
#                     query_embedding=embedding,
#                     top_k=top_k_per_collection
#                 )
                
#                 # Tag chunks with source collection for transparency
#                 for chunk in results:
#                     chunk["source_collection"] = collection_name
                
#                 all_chunks.extend(results)
#                 print(f"[INFO] Retrieved {len(results)} chunks from {collection_name}")
                
#             except Exception as e:
#                 print(f"[WARN] Failed to retrieve from {collection_name}: {e}")
#                 continue

#         # Re-rank and select top chunks across all collections
#         all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
#         state["retrieved_chunks"] = all_chunks[:5]  # Keep top 5 overall

#         print(f"[INFO] Total chunks retrieved: {len(all_chunks)}, using top 5")
        
#         return state

#     except Exception as e:
#         print(f"[ERROR] Retrieval failed: {e}")
#         state["retrieved_chunks"] = []
#         return state
    
# # --- Node 4: Context-based Generator ---
# def generate_node(state: RAGState) -> RAGState:
#     try:
#         context_parts = []
#         citations = []
#         sources = []

#         for chunk in state.get("retrieved_chunks", []):
#             meta = chunk["metadata"]
#             section_no = meta.get("section_no", "")
#             content = chunk["content"]
#             source_coll = chunk.get("source_collection", "unknown")
#             context_parts.append(f"Section {section_no}: {content}")
#             citations.append(section_no)

#             # Format based on collection type
#             if "ipc_sections" in source_coll:
#                 section_no = meta.get("section_no", "")
#                 context_parts.append(f"[IPC Section {section_no}]: {content}")
#                 citations.append(f"IPC {section_no}")
#             else:
#                 title = meta.get("title", "Legal Reference")
#                 context_parts.append(f"[{title}]: {content}")
#                 if meta.get("citations"):
#                     citations.extend(meta["citations"])
            
#             sources.append(source_coll)

#         context_str = "\n".join(context_parts) or "No external context used."

#         example_query = ("I got engaged in June 2018 and married in February 2019. Soon after, my husband stopped caring for me and the household, then left. "
#                          "He abused me for talking to friends and about my past, which he already knew. Yesterday, he called me to meet, and I hoped we could reconcile. "
#                          "Instead, he beat me and stopped me from leaving. I escaped this morning. I want a divorce as soon as possible.")
    
#         example_responses = [
#             "First, lodge an FIR under Section 498A for cruelty. Then consult a lawyer to file two cases: one under the Domestic Violence Act, and another for Judicial Separation under Section 10 of the Hindu Marriage Act, since you can’t seek divorce before one year of marriage.",
#             "You can file for divorce in family court under the Hindu Marriage Act on grounds of mental and physical cruelty. Also, register an FIR under Sections 498A and 323 IPC.",
#             "Since you married in February 2019, you must wait one year before filing for divorce. Meanwhile, file a police complaint for assault — it will support your case for cruelty. You can also claim maintenance under Section 125 CrPC if he isn’t supporting you."
#         ]
    
#         prompt = (
#             "You are a senior Indian law analyst. Follow the guidelines strictly. "
#             "Explain legal issues in simple, plain English that anyone can understand. Use a warm, professional tone, and avoid robotic phrasing. "
#             "Do not include reasoning scaffolds, JSON, or any pre/post text.\n\n"
#             "Guidelines:\n"
#             "- Internally reason as Issue → Rule (statute/precedent) → Application → Conclusion, but only output the detailed final answer.\n"
#             "- Prefer authoritative Indian sources and cite succinctly, e.g., (IPC s.498A), (HMA 1955 s.13), (CrPC s.125).\n"
#             "- If a precise section is uncertain, mention it briefly without guessing.\n"
#             "- Give concise response to ensure a Flesch Reading Ease score of atleast 55+.\n"
#             "- Explain legal terms briefly in everyday language when necessary.\n"
#             "- Give practical guidance wherever possible, focusing on what a person can realistically do.\n\n"
#             "Style Example (tone only, not ground truth):\n"
#             f"Query:\n{example_query}\n\n"
#             "Example Responses:\n"
#             f"1. {example_responses[0]}\n"
#             f"2. {example_responses[1]}\n"
#             f"3. {example_responses[2]}\n"

#             f"Context:\n{context_str}\n\n"
#             "Now answer this user query with the detailed final answer only:\n"
#             f"User Question: {state['user_query']}\n\n"
#             "Answer:"
#         )

#         generator = load_llm()
#         gen = generator(prompt, max_new_tokens=512)[0]["generated_text"]

#         state["response"] = gen
#         state["citations"] = list(set(citations))
#         state["response"] += f"\n\n_Sources: {', '.join(set(sources))}_"
#         return state

#     except Exception as e:
#         print(f"[ERROR] Generation failed: {e}")
#         state["response"] = "Sorry, I couldn't generate a response due to an internal error."
#         state["citations"] = []
#         return state


# # --- Node 5: Direct Generator (No Retrieval) ---
# def direct_generate_node(state: RAGState) -> RAGState:
#     try:
#         generator = load_llm()
#         prompt = (
#             "You are an intelligent conversational assistant with general world knowledge. "
#             "Respond helpfully, concisely, and politely.\n\n"
#             f"User: {state['user_query']}\nAssistant:"
#         )
#         gen = generator(prompt, max_new_tokens=256)[0]["generated_text"]

#         state["response"] = gen
#         state["citations"] = []
#         state["target_collections"] = []
#         return state

#     except Exception as e:
#         print(f"[ERROR] Direct generation failed: {e}")
#         state["response"] = "Sorry, I couldn't process your question right now."
#         state["citations"] = []
#         return state


# rag_graph = StateGraph(RAGState)

# rag_graph.add_node("retrieval_decision", retrieval_decision_node)
# rag_graph.add_node("domain_routing", domain_routing_node)      
# rag_graph.add_node("retrieve", retrieve_node)
# rag_graph.add_node("generate", generate_node)
# rag_graph.add_node("direct_generate", direct_generate_node)

# rag_graph.set_entry_point("retrieval_decision")
# rag_graph.add_conditional_edges(
#     "retrieval_decision",
#     lambda state: "domain_routing" if state.get("retrieval_needed") else "direct_generate"
# )
# rag_graph.add_edge("domain_routing", "retrieve")
# rag_graph.add_edge("retrieve", "generate")
# rag_graph.add_edge("generate", END)
# rag_graph.add_edge("direct_generate", END)

# app = rag_graph.compile()



"""
Adaptive RAG Graph
------------------
Enhanced RAG pipeline with query-aware adaptive retrieval.

Flow:
1. Retrieval Decision (Do we need retrieval?)
2. Domain Routing (Which collections?)
3. Query Analysis (What's the query type?)
4. Adaptive Retrieval (Auto-select fusion method)
5. LLM Generation (With query-aware context)
"""

from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from src.nodes.retrieval_decision_agent import RetrievalDecisionAgent
from src.nodes.domain_routing_agent import DomainRoutingAgent
from src.retriever.adaptive_hybrid_retriever import AdaptiveHybridRetriever
from src.retriever.query_analyzer import QueryCharacteristics
from src.llm.llm_client import load_llm
from src.config.settings import settings


# ===============================================================
# Enhanced State Definition
# ===============================================================
class AdaptiveRAGState(TypedDict):
    """State with query analysis and adaptive retrieval metadata."""
    user_query: str
    retrieval_needed: bool
    target_domains: Dict[str, float]
    target_collections: List[str]
    query_characteristics: Optional[QueryCharacteristics]  # NEW
    retrieval_explanation: str                              # NEW
    retrieved_chunks: List[dict]
    response: str
    citations: List[str]
    routing_explanation: str


# ===============================================================
# Node 1: Retrieval Decision
# ===============================================================
def retrieval_decision_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Determine if external legal knowledge is needed."""
    try:
        agent = RetrievalDecisionAgent()
        decision = agent.analyze_query(state["user_query"])
        state["retrieval_needed"] = decision["retrieval_needed"]
        print(f"[DECISION] Retrieval: {decision['retrieval_needed']}")
        return state
    except Exception as e:
        print(f"[ERROR] Retrieval decision failed: {e}")
        state["retrieval_needed"] = True
        return state


# ===============================================================
# Node 2: Domain Routing
# ===============================================================
def domain_routing_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Route query to appropriate legal domain collections."""
    try:
        router = DomainRoutingAgent()
        
        domain_scores = router.classify_domain(
            query=state["user_query"],
            threshold=0.25,
            top_k=3
        )
        
        collections_with_scores = router.route_to_collections(
            query=state["user_query"],
            threshold=0.25,
            max_collections=3
        )
        
        target_collections = [coll for coll, _ in collections_with_scores]
        explanation = router.get_routing_explanation(state["user_query"])
        
        state["target_domains"] = domain_scores
        state["target_collections"] = target_collections
        state["routing_explanation"] = explanation["explanation"]
        
        print(f"[ROUTING] Collections: {target_collections}")
        return state
        
    except Exception as e:
        print(f"[ERROR] Domain routing failed: {e}")
        state["target_domains"] = {"IPC Sections": 0.5}
        state["target_collections"] = ["ipc_sections"]
        state["routing_explanation"] = "Fallback routing due to error"
        return state


# ===============================================================
# Node 3: Adaptive Hybrid Retrieval (NEW)
# ===============================================================
def adaptive_retrieve_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """
    Query-aware adaptive retrieval with automatic fusion selection.
    """
    try:
        retriever = AdaptiveHybridRetriever(enable_analytics=True)
        
        # Adaptive retrieval
        results, characteristics = retriever.retrieve(
            query=state["user_query"],
            collection_names=state["target_collections"],
            top_k=5
        )
        
        # Update state
        state["retrieved_chunks"] = results
        state["query_characteristics"] = characteristics
        state["retrieval_explanation"] = (
            f"Query classified as '{characteristics.query_type.value}'. "
            f"Using {characteristics.recommended_fusion} fusion with "
            f"semantic weight {characteristics.recommended_weights[0]:.1f}"
        )
        
        print(f"[ADAPTIVE] Retrieved {len(results)} chunks")
        print(f"[ADAPTIVE] Strategy: {characteristics.recommended_fusion}")
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Adaptive retrieval failed: {e}")
        state["retrieved_chunks"] = []
        state["retrieval_explanation"] = "Retrieval failed"
        return state


# ===============================================================
# Node 4: Query-Aware Generation (ENHANCED)
# ===============================================================
def adaptive_generate_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """
    Generate response with query-type awareness.
    """
    try:
        # Build context from retrieved chunks
        context_parts = []
        citations = []
        sources = []
        
        for chunk in state.get("retrieved_chunks", []):
            meta = chunk["metadata"]
            content = chunk["content"]
            source_coll = chunk.get("source_collection", "unknown")
            
            # Format based on collection
            if "ipc_sections" in source_coll:
                section_no = meta.get("section_no", "")
                context_parts.append(f"[IPC Section {section_no}]: {content}")
                citations.append(f"IPC {section_no}")
            else:
                title = meta.get("title", "Legal Reference")
                context_parts.append(f"[{title}]: {content}")
                if meta.get("citations"):
                    citations.extend(meta["citations"])
            
            sources.append(source_coll)
        
        context_str = "\n\n".join(context_parts) or "No specific legal provisions found."
        
        # Get query characteristics for context-aware prompting
        query_char = state.get("query_characteristics")
        query_type = query_char.query_type.value if query_char else "general"
        
        # Adapt prompt based on query type
        prompt = _build_adaptive_prompt(
            query=state["user_query"],
            context=context_str,
            query_type=query_type,
            routing_info=state.get("routing_explanation", ""),
            retrieval_info=state.get("retrieval_explanation", "")
        )
        
        # Generate response
        generator = load_llm()
        gen = generator(prompt, max_new_tokens=512)[0]["generated_text"]
        
        state["response"] = gen
        state["citations"] = list(set(citations))
        
        # Add metadata footer
        state["response"] += f"\n\n_Query Type: {query_type} | Sources: {', '.join(set(sources))}_"
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        state["response"] = "Unable to generate response due to internal error."
        state["citations"] = []
        return state


def _build_adaptive_prompt(
    query: str,
    context: str,
    query_type: str,
    routing_info: str,
    retrieval_info: str
) -> str:
    """
    Build query-type-specific prompts for better responses.
    """
    base_instructions = (
        "You are a senior Indian legal analyst. Follow these core principles:\n"
        "- Use plain English (Flesch Reading Ease 55+)\n"
        "- Cite specific sections/acts succinctly\n"
        "- Provide practical guidance\n"
        "- Acknowledge uncertainties\n\n"
    )
    
    # Type-specific instructions
    type_instructions = {
        "exact_match": (
            "The user is looking for a specific legal provision. "
            "Provide: (1) Full text of the section, (2) Brief explanation, (3) Relevant penalties/procedures.\n"
        ),
        "conceptual": (
            "The user seeks understanding of a legal concept. "
            "Provide: (1) Clear definition, (2) Key principles, (3) Practical examples.\n"
        ),
        "procedural": (
            "The user wants to know how to do something legally. "
            "Provide: (1) Step-by-step process, (2) Required documents, (3) Timeline expectations.\n"
        ),
        "case_based": (
            "The user has a specific situation. "
            "Provide: (1) Applicable law, (2) User's rights/options, (3) Recommended next steps.\n"
        ),
        "multi_aspect": (
            "The query spans multiple legal areas. "
            "Provide: (1) Break down each aspect, (2) Cross-domain implications, (3) Prioritized action plan.\n"
        )
    }
    
    specific_instruction = type_instructions.get(query_type, type_instructions["conceptual"])
    
    prompt = f"""
{base_instructions}

**Query Analysis:**
{retrieval_info}

**Identified Legal Domains:**
{routing_info}

**Query-Specific Guidance:**
{specific_instruction}

**Legal Context:**
{context}

**User Question:**
{query}

**Detailed Answer:**
"""
    
    return prompt


# ===============================================================
# Node 5: Direct Generation (No Retrieval)
# ===============================================================
def direct_generate_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Handle queries that don't need legal retrieval."""
    try:
        generator = load_llm()
        prompt = (
            "You are a helpful assistant. Respond naturally and concisely.\n\n"
            f"User: {state['user_query']}\n"
            "Assistant:"
        )
        gen = generator(prompt, max_new_tokens=256)[0]["generated_text"]
        
        state["response"] = gen
        state["citations"] = []
        state["target_collections"] = []
        return state
        
    except Exception as e:
        print(f"[ERROR] Direct generation failed: {e}")
        state["response"] = "Sorry, I couldn't process your question."
        state["citations"] = []
        return state


# ===============================================================
# Build Adaptive RAG Graph
# ===============================================================
adaptive_rag_graph = StateGraph(AdaptiveRAGState)

# Add nodes
adaptive_rag_graph.add_node("retrieval_decision", retrieval_decision_node)
adaptive_rag_graph.add_node("domain_routing", domain_routing_node)
adaptive_rag_graph.add_node("adaptive_retrieve", adaptive_retrieve_node)  # NEW
adaptive_rag_graph.add_node("adaptive_generate", adaptive_generate_node)  # UPDATED
adaptive_rag_graph.add_node("direct_generate", direct_generate_node)

# Define flow
adaptive_rag_graph.set_entry_point("retrieval_decision")

adaptive_rag_graph.add_conditional_edges(
    "retrieval_decision",
    lambda state: "domain_routing" if state.get("retrieval_needed") else "direct_generate"
)

adaptive_rag_graph.add_edge("domain_routing", "adaptive_retrieve")
adaptive_rag_graph.add_edge("adaptive_retrieve", "adaptive_generate")
adaptive_rag_graph.add_edge("adaptive_generate", END)
adaptive_rag_graph.add_edge("direct_generate", END)

# Compile
app = adaptive_rag_graph.compile()


# ===============================================================
# Test Adaptive RAG
# ===============================================================
if __name__ == "__main__":
    test_queries = [
        # Different query types to test adaptation
        "Section 498A IPC domestic violence",
        "What is the concept of bail in Indian law?",
        "How do I file for divorce in India?",
        "My employer terminated me without notice, what are my rights?",
        "Landlord eviction and false criminal case - need help"
    ]
    
    print("\n" + "="*80)
    print("ADAPTIVE RAG SYSTEM - COMPREHENSIVE TEST")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}\n")
        
        initial_state = {
            "user_query": query,
            "retrieval_needed": False,
            "target_domains": {},
            "target_collections": [],
            "query_characteristics": None,
            "retrieval_explanation": "",
            "retrieved_chunks": [],
            "response": "",
            "citations": [],
            "routing_explanation": ""
        }
        
        final_state = app.invoke(initial_state)
        
        print("\n[ANALYSIS]")
        print(f"  Routing: {final_state.get('routing_explanation', 'N/A')}")
        print(f"  Retrieval: {final_state.get('retrieval_explanation', 'N/A')}")
        
        if final_state.get("query_characteristics"):
            qc = final_state["query_characteristics"]
            print(f"  Query Type: {qc.query_type.value}")
            print(f"  Complexity: {qc.complexity_score:.2f}")
        
        print("\n[RESPONSE]")
        print(final_state.get("response", "No response"))
        
        if final_state.get("citations"):
            print(f"\n[CITATIONS] {', '.join(final_state['citations'])}")
        
        input("\nPress Enter for next query...")