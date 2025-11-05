import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]  # Go up two levels: src/vector_store -> project root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.vector_store.milvus_store import connect_milvus

# Connect to Milvus
client = connect_milvus()

# List all existing collections
collections = client.list_collections()

print("📚 Available Collections in Milvus:")
for c in collections:
    print(" -", c)



# for name in client.list_collections():
#     stats = client.get_collection_stats(collection_name=name)
#     print(f"\n📂 Collection: {name}")
#     print(f"   Number of Entities: {stats['row_count']}")



# Drop collection
# # client.drop_collection(collection_name="civil_store")
# # print("🗑️ Deleted collection 'civil_law_store'")
