"""
LLM Client + RAG Response Generator
-----------------------------------
Handles:
1. Hugging Face LLM loading (Llama-3.1-8B-Instruct)
2. Query embedding + retrieval from Milvus
3. Context-aware response generation
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.vector_store.milvus_store import search_similar_chunks, connect_milvus, load_collection
from src.config.settings import settings
from src.retriever.retriever import get_embedding
from huggingface_hub import login


# ===============================================================
# 1️⃣ Load Hugging Face Model
# ===============================================================
login(token=settings.HF_API_KEY)

def load_llm():
    """
    Loads the Llama-3.1-8B-Instruct model with an optimized text-generation pipeline.
    """
    print(" Loading Llama-3.1-8B-Instruct model...")

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" ,   # Utilizes available GPU if present
        
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.1
    )

    print("✅ LLM loaded successfully.")
    return generator


# # ===============================================================
# # 2️⃣ Load Embedding Model (for query encoding)
# # ===============================================================
# def load_embedding_model():
#     """
#     Loads a small, efficient embedding model for semantic retrieval.
#     """
#     print("⏳ Loading embedding model (all-MiniLM-L6-v2)...")
#     embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     print("✅ Embedding model ready.")
#     return embedder


# ===============================================================
# 3️⃣ Generate Response (Full RAG flow)
# ===============================================================
def generate_rag_response(user_query: str, top_k: int = 3):
    """
    Retrieves relevant IPC sections and generates a final answer using Llama-3.1-8B.

    Steps:
    1. Convert query → embedding
    2. Retrieve top-K similar chunks from Milvus
    3. Build context prompt
    4. Generate final response
    """
    print("\n🔍 Processing query through RAG pipeline...")

    # -- Connect to Milvus
    client = connect_milvus()
    load_collection(client, settings.MILVUS_COLLECTION)

    # -- Encode query
    query_embedding = get_embedding(user_query)

    # -- Retrieve relevant chunks
    results = search_similar_chunks(client, settings.MILVUS_COLLECTION, query_embedding, top_k=top_k)

    # -- Build context
    context = ""
    citations = []
    for r in results:
        meta = r["metadata"]
        context += f"\nSection {meta.get('section_no', '')}: {r['content']}\n"
        citations.append(meta.get('section_no', 'Unknown'))

    # -- Prompt construction
    prompt = f"""
You are a Legal Assistant AI specializing in the Indian Penal Code (IPC 1860).
Answer the user's query clearly and accurately using the following relevant sections.
Always cite section numbers when applicable.

Context:
{context}

User Query:
{user_query}

Answer:
"""

    # -- Generate response
    generator = load_llm()
    response = generator(prompt, max_new_tokens=300)[0]['generated_text']

    print("✅ RAG response generated successfully.\n")
    return {
        "response": response,
        "citations": list(set(citations))
    }
