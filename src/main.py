from src.rag.rag_graph import app

initial_state = {
    "user_query": "What is the punishment for section 1 of IPC?",
    "query_embedding": [],
    "retrieved_chunks": [],
    "response": "",
    "citations": []
}

final_state = app.invoke(initial_state)
print("Answer:", final_state["response"])
print("Cited sections:", final_state["citations"])
