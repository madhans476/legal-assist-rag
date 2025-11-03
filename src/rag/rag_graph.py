# src/rag/rag_graph.py

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from src.retriever.retriever import get_embedding
from src.vector_store.milvus_store import connect_milvus, load_collection, search_similar_chunks
from src.llm.llm_client import load_llm
from src.config.settings import settings

class RAGState(TypedDict):
    user_query: str
    query_embedding: List[float]
    retrieved_chunks: List[dict]
    response: str
    citations: List[str]

def retrieve_node(state: RAGState) -> RAGState:
    embedding = get_embedding(state["user_query"])
    state["query_embedding"] = embedding

    client = connect_milvus()
    load_collection(client=client, collection_name=settings.MILVUS_COLLECTION)

    top_k = 5
    results = search_similar_chunks(
        client=client,
        collection_name=settings.MILVUS_COLLECTION,
        query_embedding=embedding,
        top_k=top_k
    )

    state["retrieved_chunks"] = results
    return state

def generate_node(state: RAGState) -> RAGState:
    context_parts = []
    citations = []
    for chunk in state["retrieved_chunks"]:
        meta = chunk["metadata"]
        section_no = meta.get("section_no", "")
        content = chunk["content"]
        context_parts.append(f"Section {section_no}: {content}")
        citations.append(section_no)

    context_str = "\n".join(context_parts)

    example_query = ("I got engaged in June 2018 and married in February 2019. Soon after, my husband stopped caring for me and the household, then left. "
                     "He abused me for talking to friends and about my past, which he already knew. Yesterday, he called me to meet, and I hoped we could "
                     "reconcile. Instead, he beat me and stopped me from leaving. I escaped this morning. I want a divorce as soon as possible.")

    example_responses = [
        "First, lodge an FIR under Section 498A for cruelty. Then consult a lawyer to file two cases: one under the Domestic Violence Act, and another for Judicial Separation under Section 10 of the Hindu Marriage Act, since you can’t seek divorce before one year of marriage.",
        "You can file for divorce in family court under the Hindu Marriage Act on grounds of mental and physical cruelty. Also, register an FIR under Sections 498A and 323 IPC.",
        "Since you married in February 2019, you must wait one year before filing for divorce. Meanwhile, file a police complaint for assault — it will support your case for cruelty. You can also claim maintenance under Section 125 CrPC if he isn’t supporting you."
    ]

    prompt = (
        "You are a senior Indian law analyst. Follow the guidelines strictly. "
        "Explain legal issues in simple, plain English that anyone can understand. Use a warm, professional tone, and avoid robotic phrasing. "
        "Do not include reasoning scaffolds, JSON, or any pre/post text.\n\n"
        "Guidelines:\n"
        "- Internally reason as Issue → Rule (statute/precedent) → Application → Conclusion, but only output the detailed final answer.\n"
        "- Prefer authoritative Indian sources and cite succinctly, e.g., (IPC s.498A), (HMA 1955 s.13), (CrPC s.125).\n"
        "- If a precise section is uncertain, mention it briefly without guessing.\n"
        "- Give concise response to ensure a Flesch Reading Ease score of atleast 55+.\n"
        "- Explain legal terms briefly in everyday language when necessary.\n"
        "- Give practical guidance wherever possible, focusing on what a person can realistically do.\n\n"
        "Style Example (tone only, not ground truth):\n"
        f"Query:\n{example_query}\n\n"
        "Example Responses:\n"
        f"1. {example_responses[0]}\n"
        f"2. {example_responses[1]}\n"
        f"3. {example_responses[2]}\n"
        f"Context:\n{context_str}\n\n"
        "Now answer this user query with the detailed final answer only:\n"
        f"Question: {state['user_query']}\n\n"
        "Answer:"
    )

    generator = load_llm()
    gen = generator(prompt, max_new_tokens=512)[0]["generated_text"]

    state["response"] = gen
    state["citations"] = list(set(citations))
    return state

rag_graph = StateGraph(RAGState)
rag_graph.add_node("retrieve", retrieve_node)
rag_graph.add_node("generate", generate_node)

rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "generate")
rag_graph.add_edge("generate", END)

app = rag_graph.compile()
