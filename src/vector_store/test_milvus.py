from src.vector_store.milvus_store import connect_milvus, create_collection, insert_chunks
from src.retriever.retriever import build_retriever_pipeline
from src.config.settings import settings

# 1. Connect to Milvus
client = connect_milvus()

# 2. Create or load collection
create_collection(client, settings.MILVUS_COLLECTION)

# 3. Build retriever pipeline and insert
chunks = build_retriever_pipeline()
insert_chunks(client, settings.MILVUS_COLLECTION, chunks)