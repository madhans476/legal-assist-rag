# test_retriever.py
from src.retriever.retriever import build_retriever_pipeline

chunks = build_retriever_pipeline()
print(chunks[0].keys())  # -> dict_keys(['content', 'metadata', 'embedding'])
print(chunks[0]["metadata"])
print(f"Lenght: {len(chunks)}")
print(chunks[0])
print(chunks[7])
