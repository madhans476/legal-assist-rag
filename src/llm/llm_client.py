# src/llm/llm_client.py

"""
LLM Client + RAG Response Generator
-----------------------------------
Handles:
1. Hugging Face LLM loading (Llama-3.1-8B-Instruct)
2. Query embedding + retrieval from Milvus
3. Context-aware response generation
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from src.vector_store.milvus_store import search_similar_chunks, connect_milvus, load_collection
from src.config.settings import settings
from src.retriever.retriever import get_embedding

# Login to Hugging Face
login(token=settings.HF_API_KEY)

# Choose a load profile:
#   "gpu_4bit"   -> for NVIDIA GPU with limited VRAM (recommended if you have CUDA)
#   "cpu"        -> for CPU-only inference (slow but simple)
#   "disk_offload" -> for explicit NVMe offload when RAM/VRAM are insufficient
LOAD_PROFILE = os.environ.get("LLM_LOAD_PROFILE", "gpu_4bit").lower()

def _load_llm_gpu_4bit(model_name: str):
    """
    Load with 4-bit quantization + device_map=auto and a VRAM cap.
    Requires: pip install bitsandbytes, CUDA available.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Cap VRAM per GPU if needed (adjust "10GB" based on your GPU, e.g., 12GB/22GB)
    # If you have 2x T4, split like {"0": "12GB", "1": "12GB"} or similar.
    max_memory = {"0": "10GB"}  # change per your GPU capacity

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        max_memory=max_memory,
        torch_dtype="auto",  # warning about torch_dtype deprecation is benign; can switch to dtype in newer transformers
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.1,
    )

def _load_llm_cpu(model_name: str):
    """
    Force CPU-only load to avoid Accelerate dispatch and disk offload errors.
    Slow, but robust on Windows without CUDA.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"},
        torch_dtype="auto",
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.1,
    )

def _load_llm_disk_offload(model_name: str):
    """
    Explicit NVMe disk offload using Accelerate utilities.
    Use when both GPU and CPU RAM are insufficient.
    Requires a fast SSD/NVMe and accelerate>=0.26.
    """
    from accelerate import disk_offload  # part of HF Accelerate
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    offload_dir = os.path.abspath("./offload_dir")
    os.makedirs(offload_dir, exist_ok=True)
    # Put execution on CPU (can use "cuda:0" if you have some VRAM)
    disk_offload(model, offload_dir=offload_dir, execution_device="cpu")
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.1,
    )

def load_llm():
    """
    Loads the Llama-3.1-8B-Instruct model with an optimized text-generation pipeline.
    Automatically selects a load profile to avoid 'disk offload' ValueError.
    """
    print("⏳ Loading Llama-3.1-8B-Instruct model...")
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    try:
        if LOAD_PROFILE == "gpu_4bit":
            generator = _load_llm_gpu_4bit(model_name)
        elif LOAD_PROFILE == "disk_offload":
            generator = _load_llm_disk_offload(model_name)
        else:
            generator = _load_llm_cpu(model_name)
    except ValueError as e:
        # Fallbacks if device_map="auto" attempts full disk offload implicitly
        msg = str(e)
        if "offload the whole model to the disk" in msg:
            print("⚠️ Implicit full disk offload detected. Falling back to explicit disk_offload profile.")
            generator = _load_llm_disk_offload(model_name)
        else:
            print("⚠️ Error while loading LLM; falling back to CPU-only.")
            generator = _load_llm_cpu(model_name)

    print("✅ LLM loaded successfully.")
    return generator

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

    # -- Prompt
    prompt = f"""
You are a Legal Assistant AI specializing in the Indian Penal Code (IPC 1860).
Answer the user's query clearly and accurately using the following relevant sections.
Always cite section numbers when applicable.

Context:
{context}

User Query:
{user_query}

Answer:
""".strip()

    # -- Generate
    generator = load_llm()
    response = generator(prompt, max_new_tokens=300)[0]['generated_text']

    print("✅ RAG response generated successfully.\n")
    return {
        "response": response,
        "citations": list(set(citations))
    }
