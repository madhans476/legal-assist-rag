"""
Legal Assist RAG - FastAPI Server
==================================
Production-grade API server for Legal RAG chatbot with streaming support.

Features:
- Streaming responses via Server-Sent Events (SSE)
- Thread-based conversation history
- Query analysis and adaptive retrieval
- Follow-up question generation
- Legal domain routing
- Comprehensive error handling
"""

import os
import json
import uvicorn
import traceback
from pathlib import Path
from datetime import datetime
from typing import Annotated, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_mistralai.chat_models import ChatMistralAI

# Import your RAG components
from src.rag.rag_graph import app as rag_app
from src.config.settings import settings

# --- Configuration ---
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

# if "MISTRAL_API_KEY" not in os.environ:
#     raise EnvironmentError("MISTRAL_API_KEY not found in .env file")

# --- Initialize LLM for follow-up generation ---
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)

# --- System Prompt ---
SYSTEM_PROMPT = """You are "Legal Assist AI" - an expert Indian legal assistant specializing in providing accurate, actionable legal guidance.

**IMPORTANT - Domain Restriction:**
- You specialize in Indian law: IPC, CrPC, CPC, Constitutional Law, Family Law, Property Law, Labour Law, Business Law, and related legal topics.
- For non-legal queries (general knowledge, entertainment, etc.), politely decline and redirect to legal topics.
- Example: "I specialize in Indian legal matters. I can help with criminal law, civil disputes, family law, property issues, and more. Do you have a legal question?"

**Core Principles:**
- Provide clear, practical legal guidance in plain English
- Always cite specific sections/articles when applicable (e.g., IPC s.420, CrPC s.125)
- Explain legal terms in simple language
- Focus on actionable steps the user can take
- Acknowledge uncertainties when precise legal advice requires a lawyer

**Response Guidelines:**
1. Structure: Use markdown headings (###), bold (**text**), and lists for clarity
2. Math/Formulas: Use LaTeX with `$...$` (inline) or `$$ ... $$` (display)
3. Code: Only provide code snippets if explicitly requested
4. Tables: Use for comparisons, feature lists, or structured data
5. Tone: Warm, professional, and supportive - like a knowledgeable legal advisor

**Legal Citation Format:**
- Sections: "Section 420 IPC" or "(IPC s.420)"
- Articles: "Article 21 of Constitution" or "(Constitution Art.21)"
- Acts: "Hindu Marriage Act 1955 s.13" or "(HMA 1955 s.13)"

**Response Style:**
- Flesch Reading Ease score of 55+ (clear, accessible language)
- Avoid legalese unless necessary, then explain it
- Break complex procedures into numbered steps
- Provide realistic timelines when discussing legal processes
"""

# --- History Management ---
def get_history_path(thread_id: str) -> Path:
    return HISTORY_DIR / f"{thread_id}.json"

def load_history(thread_id: str) -> dict:
    """Load conversation history for a thread."""
    path = get_history_path(thread_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading history for {thread_id}: {e}")
    return {
        "thread_id": thread_id,
        "messages": [],
        "created_at": datetime.now().isoformat()
    }

def save_history(thread_id: str, messages: list):
    """Save conversation history for a thread."""
    path = get_history_path(thread_id)
    history = load_history(thread_id)

    serializable_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            serializable_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serializable_messages.append({"role": "assistant", "content": msg.content})

    history["messages"] = serializable_messages
    history["updated_at"] = datetime.now().isoformat()

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error saving history for {thread_id}: {e}")

def list_all_threads() -> list:
    """List all conversation threads with metadata."""
    threads = []
    for file_path in HISTORY_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                preview_text = ""
                if data.get("messages"):
                    first_msg = data["messages"][0]
                    preview_text = first_msg.get("content", "")[:100]
                
                threads.append({
                    "thread_id": data.get("thread_id", file_path.stem),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                    "preview": preview_text
                })
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    threads.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return threads

def delete_thread(thread_id: str) -> bool:
    """Delete a conversation thread."""
    path = get_history_path(thread_id)
    try:
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception as e:
        print(f"❌ Error deleting thread {thread_id}: {e}")
        return False

# --- API Models ---
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's legal question")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    citations: Optional[list] = []
    follow_up: Optional[str] = None
    routing_info: Optional[str] = None

class ThreadInfo(BaseModel):
    thread_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int
    preview: str

class HistoryResponse(BaseModel):
    thread_id: str
    messages: list
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Legal Assist RAG API starting up...")
    print(f"📁 History directory: {HISTORY_DIR.absolute()}")
    print(f"📊 Milvus Host: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    yield
    print("👋 Legal Assist RAG API shutting down...")

# --- FastAPI App ---
app = FastAPI(
    title="Legal Assist RAG API",
    description="Production-grade Indian legal assistant with adaptive RAG and streaming",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Legal Assist RAG API",
        "version": "1.0.0",
        "capabilities": [
            "Indian Legal Query Processing",
            "Adaptive Hybrid Retrieval",
            "Multi-domain Routing",
            "Streaming Responses",
            "Conversation History"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "milvus_configured": bool(settings.MILVUS_HOST),
        "history_enabled": HISTORY_DIR.exists()
    }

# --- Streaming Chat Endpoint (PRIMARY) ---
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint that returns responses token-by-token via SSE.
    Includes legal domain routing, adaptive retrieval, and follow-up generation.
    """
    try:
        if not request.thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required")

        async def generate():
            try:
                # Load history
                history_data = load_history(request.thread_id)
                existing_messages = []
                for msg in history_data.get("messages", []):
                    if msg["role"] == "user":
                        existing_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        existing_messages.append(AIMessage(content=msg["content"]))

                # Prepare RAG state
                initial_state = {
                    "user_query": request.query,
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

                # Signal start of processing
                yield f"data: {json.dumps({'type': 'processing', 'message': 'Analyzing query...'})}\n\n"

                # Run RAG graph
                final_state = rag_app.invoke(initial_state)

                # Extract response
                full_response = final_state.get("response", "")
                citations = final_state.get("citations", [])
                routing_info = final_state.get("routing_explanation", "")

                # Stream the response token by token (simulate streaming)
                # In production, you might want to integrate actual LLM streaming
                words = full_response.split()
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                # Generate follow-up question
                follow_up_text = ""
                try:
                    follow_prompt = f"""Based on this legal conversation:

User asked: "{request.query}"
Assistant provided legal guidance about: {full_response[:200]}...

Generate ONE concise, engaging follow-up question (max 15 words) that:
- Encourages deeper exploration of the legal issue
- Is specific and actionable
- Feels natural and helpful

Provide ONLY the question text, nothing else."""

                    follow_messages = [
                        SystemMessage(content="You are a helpful assistant that generates engaging legal follow-up questions."),
                        HumanMessage(content=follow_prompt)
                    ]
                    
                    follow_response = llm.invoke(follow_messages)
                    follow_up_text = follow_response.content.strip()
                    
                    # Clean up
                    follow_up_text = follow_up_text.strip('"\'').strip()
                    if follow_up_text and follow_up_text[0].isdigit() and '.' in follow_up_text[:3]:
                        follow_up_text = follow_up_text.split('.', 1)[1].strip()
                    
                except Exception as e:
                    print(f"⚠️ Follow-up generation failed: {e}")

                # Save history
                new_user_message = HumanMessage(content=request.query)
                new_ai_message = AIMessage(content=full_response)
                all_messages = existing_messages + [new_user_message, new_ai_message]
                save_history(request.thread_id, all_messages)

                # Signal completion
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
                # Send citations if available
                if citations:
                    yield f"data: {json.dumps({'type': 'citations', 'content': citations})}\n\n"
                
                # Send routing info if available
                if routing_info:
                    yield f"data: {json.dumps({'type': 'routing', 'content': routing_info})}\n\n"
                
                # Send follow-up
                if follow_up_text:
                    yield f"data: {json.dumps({'type': 'follow_up', 'content': follow_up_text})}\n\n"
                
            except Exception as e:
                print(f"❌ Streaming error: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except Exception as e:
        print(f"❌ Error in chat/stream endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Non-streaming endpoint (for compatibility) ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming endpoint for legal queries."""
    try:
        if not request.thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required")

        # Load history
        history_data = load_history(request.thread_id)
        existing_messages = []
        for msg in history_data.get("messages", []):
            if msg["role"] == "user":
                existing_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                existing_messages.append(AIMessage(content=msg["content"]))

        # Prepare RAG state
        initial_state = {
            "user_query": request.query,
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

        # Run RAG graph
        final_state = rag_app.invoke(initial_state)

        # Extract results
        ai_answer = final_state.get("response", "")
        citations = final_state.get("citations", [])
        routing_info = final_state.get("routing_explanation", "")

        # Generate follow-up
        follow_up_text = ""
        try:
            follow_prompt = f"""Based on this legal conversation:

User asked: "{request.query}"
Assistant provided legal guidance about: {ai_answer[:200]}...

Generate ONE concise, engaging follow-up question (max 15 words).
Provide ONLY the question text."""

            follow_messages = [
                SystemMessage(content="You are a helpful assistant that generates engaging legal follow-up questions."),
                HumanMessage(content=follow_prompt)
            ]
            follow_response = llm.invoke(follow_messages)
            follow_up_text = follow_response.content.strip().strip('"\'')
        except Exception as e:
            print(f"⚠️ Follow-up generation failed: {e}")

        # Save history
        new_user_message = HumanMessage(content=request.query)
        new_ai_message = AIMessage(content=ai_answer)
        all_messages = existing_messages + [new_user_message, new_ai_message]
        save_history(request.thread_id, all_messages)

        return ChatResponse(
            answer=ai_answer,
            thread_id=request.thread_id,
            citations=citations,
            follow_up=follow_up_text,
            routing_info=routing_info
        )
    
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads", response_model=list[ThreadInfo])
async def get_threads():
    """List all conversation threads."""
    try:
        return list_all_threads()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str):
    """Get conversation history for a specific thread."""
    try:
        return load_history(thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/threads/{thread_id}")
async def delete_conversation(thread_id: str):
    """Delete a conversation thread."""
    try:
        if delete_thread(thread_id):
            return {"status": "deleted", "thread_id": thread_id}
        raise HTTPException(status_code=404, detail="Thread not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_query(request: ChatRequest):
    """
    Analyze a query without generating a response.
    Useful for debugging and understanding query routing.
    """
    try:
        from src.retriever.query_analyzer import QueryAnalyzer
        from src.nodes.domain_routing_agent import DomainRoutingAgent
        
        analyzer = QueryAnalyzer()
        router = DomainRoutingAgent()
        
        # Analyze query
        characteristics = analyzer.analyze(request.query)
        routing = router.get_routing_explanation(request.query)
        
        return {
            "query": request.query,
            "query_type": characteristics.query_type.value,
            "complexity": characteristics.complexity_score,
            "has_legal_references": characteristics.has_legal_references,
            "recommended_fusion": characteristics.recommended_fusion,
            "recommended_weights": characteristics.recommended_weights,
            "domains": routing["identified_domains"],
            "collections": routing["target_collections"],
            "explanation": routing["explanation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Legal Assist RAG API server...")
    print("📚 Supported Legal Domains:")
    print("   - Criminal Law (IPC, CrPC)")
    print("   - Civil Law (CPC)")
    print("   - Family Law")
    print("   - Property Law")
    print("   - Labour Law")
    print("   - Business Law")
    print("   - Constitutional Law")
    print("   - Consumer Law")
    print("   - Taxation")
    print("   - Intellectual Property")
    print("\n🔗 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")



# from src.rag.rag_graph import app
# import json

# initial_state = {
#     # "user_query": "My Flat based in Tughlakabad, Delhi has been captured by one of the goon in my area. Please, suggest some of the curative actions that can be taken.",
#     # "user_query": "Hi, \nMy father received a piece of land from his mother which was passed down to him through her mother. He conducted business in this land for many years and acquired offer properties. Are these properties he acquired deemed as his self acquired property or ancestral property. Can his children file a partition suit for all the properties?",
#     "user_query": "I am thinking about setting up a public trust in Maharashtra based on Maharashtra Public Trust Act by contributing my agricultural land as a corpus fund of the trust. Here is my idea: I will act as a settlor without any rights in the management of the trust, my two friends (all of them are agriculturists) will become trustees and they will utilize the land, and the general public in the community will become beneficiaries. The purpose of the trust will be to promote people's welfare through charitable/social/educational/religious activities. After settig up the trust, I plan to transfer the land to the trust on practical base (7/12 ownership change, mutation entry) at an appropriate timing. Is this plan feasible?",
#     "query_embedding": [],
#     "retrieval_needed":True,
#     "target_domains": {},
#     "target_collections": [],
#     "retrieved_chunks": [],
#     "response": "",
#     "citations": [],
#     "routing_explanation": ""
# }

# final_state = app.invoke(initial_state)
# print("==========================================================================\n\n\n")
# # print(json.dumps(final_state, indent = 2))
# print("Answer:", final_state["response"])
# print("==========================================================================\n\n\n")
# print("Cited sections:", final_state["citations"])
