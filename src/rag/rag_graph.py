from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from src.retriever.retriever import get_embedding
from src.vector_store.milvus_store import connect_milvus, load_collection, search_similar_chunks
from src.llm.llm_client import load_llm
from src.config.settings import settings

# 1️⃣ Define Graph State
class RAGState(TypedDict):
    user_query: str
    query_embedding: List[float]
    retrieved_chunks: List[dict]  # list of dicts with content + metadata
    response: str
    citations: List[str]

# 2. Define Nodes

def retrieve_node(state: RAGState) -> RAGState:
    """
    Node: Retrieves relevant chunks for the user query via Milvus.
    Updates state with embeddings + retrieved_chunks.
    """
    # Encode the user query
    embedding = get_embedding(state["user_query"])
    state["query_embedding"] = embedding

    # Connect to Milvus and ensure collection loaded
    client = connect_milvus()
    load_collection(client=client, collection_name=settings.MILVUS_COLLECTION)  # adapt load_collection to accept client

    # Retrieve top chunks
    top_k = 3
    results = search_similar_chunks(
        client=client,
        collection_name=settings.MILVUS_COLLECTION,  # will use default from settings
        query_embedding=embedding,
        top_k=top_k
    )

    state["retrieved_chunks"] = results
    return state

def generate_node(state: RAGState) -> RAGState:
    """
    Node: Generates a response using Llama-3.1-8B-Instruct from HF.
    Uses the retrieved chunks as context, and writes the response + citations into state.
    """
    # Prepare context
    context_parts = []
    citations = []
    for chunk in state["retrieved_chunks"]:
        meta = chunk["metadata"]
        section_no = meta.get("section_no", "")
        content = chunk["content"]
        context_parts.append(f"Section {section_no}: {content}")
        citations.append(section_no)

    context_str = "\n".join(context_parts)

    prompt = (
        "You are a legal assistant knowledgeable about IPC 1860.\n"
        "Use the following relevant legal sections to answer precisely. Cite section numbers when relevant.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {state['user_query']}\n\n"
        "Answer:"
    )

    generator = load_llm()
    gen = generator(prompt, max_new_tokens=300)[0]["generated_text"]

    state["response"] = gen
    state["citations"] = list(set(citations))
    return state

# 3. Build the Graph

rag_graph = StateGraph(RAGState)
rag_graph.add_node("retrieve", retrieve_node)
rag_graph.add_node("generate", generate_node)

# Entry point: first retrieve, then generate
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "generate")
rag_graph.add_edge("generate", END)

app = rag_graph.compile()