from src.rag.rag_graph import app
import json

initial_state = {
    # "user_query": "My Flat based in Tughlakabad, Delhi has been captured by one of the goon in my area. Please, suggest some of the curative actions that can be taken.",
    # "user_query": "Hi, \nMy father received a piece of land from his mother which was passed down to him through her mother. He conducted business in this land for many years and acquired offer properties. Are these properties he acquired deemed as his self acquired property or ancestral property. Can his children file a partition suit for all the properties?",
    "user_query": "I am thinking about setting up a public trust in Maharashtra based on Maharashtra Public Trust Act by contributing my agricultural land as a corpus fund of the trust. Here is my idea: I will act as a settlor without any rights in the management of the trust, my two friends (all of them are agriculturists) will become trustees and they will utilize the land, and the general public in the community will become beneficiaries. The purpose of the trust will be to promote people's welfare through charitable/social/educational/religious activities. After settig up the trust, I plan to transfer the land to the trust on practical base (7/12 ownership change, mutation entry) at an appropriate timing. Is this plan feasible?",
    "query_embedding": [],
    "retrieval_needed":True,
    "target_domains": {},
    "target_collections": [],
    "retrieved_chunks": [],
    "response": "",
    "citations": [],
    "routing_explanation": ""
}

final_state = app.invoke(initial_state)
print("==========================================================================\n\n\n")
# print(json.dumps(final_state, indent = 2))
print("Answer:", final_state["response"])
print("==========================================================================\n\n\n")
print("Cited sections:", final_state["citations"])
