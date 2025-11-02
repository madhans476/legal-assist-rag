"""
RAG Pipeline
------------
Combines:
1. Retriever (Milvus search)
2. LLM (Hugging Face inference)
Generates contextual answers with IPC citations.
"""

from src.vector_store.milvus_store import search_similar_chunks, connect_milvus
from src.retriever.retriever import get_embedding
from src.llm.llm_client import HuggingFaceLLM
from src.config.settings import settings


def generate_legal_answer(query: str, top_k: int = 3):
    """
    Retrieves relevant IPC sections and generates an answer.
    """
    # 1️⃣ Connect Milvus
    client = connect_milvus()

    # 2️⃣ Embed the query
    query_embedding = get_embedding(query)

    # 3️⃣ Retrieve similar chunks
    results = search_similar_chunks(client, settings.MILVUS_COLLECTION, query_embedding, top_k)

    # 4️⃣ Prepare context for LLM
    context_text = ""
    citations = []
    for r in results:
        meta = r["metadata"]
        context_text += f"\nSection {meta.get('section_no', '')}: {r['content']}\n"
        citations.append(meta.get("section_no", "Unknown"))

    # 5️⃣ Build the prompt
    prompt = f"""
You are a legal assistant trained on the Indian Penal Code (IPC 1860).
Use the context below to answer the user's question precisely and cite relevant IPC sections.

Question: {query}

Context:
{context_text}

Format your answer as:
Answer: <response>
Cited Sections: <comma-separated section numbers>
"""

    # 6️⃣ Generate the response
    llm = HuggingFaceLLM()
    answer = llm.generate(prompt)

    # 7️⃣ Return final result
    return {
        "query": query,
        "answer": answer,
        "citations": citations
    }
