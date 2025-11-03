from src.rag.rag_graph import app

initial_state = {
    "user_query": "My Flat based in Tughlakabad, Delhi has been captured by one of the goon in my area. Please, suggest some of the curative actions that can be taken.",
    "query_embedding": [],
    "retrieved_chunks": [],
    "response": "",
    "citations": []
}

final_state = app.invoke(initial_state)
print("Answer:", final_state["response"])
print("Cited sections:", final_state["citations"])
