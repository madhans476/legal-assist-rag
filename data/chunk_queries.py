import os
import json
import tiktoken
from tqdm import tqdm

# Define constants
DATA_DIR = "./grouped_categories"
OUTPUT_DIR = "./chunked_categories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure tokenization
ENCODER = tiktoken.get_encoding("cl100k_base")  # same as OpenAI models
CHUNK_SIZE = 800  # tokens per chunk
CHUNK_OVERLAP = 100  # tokens overlap between chunks

def tokenize_text(text):
    """Convert text into tokens."""
    return ENCODER.encode(text)

def detokenize_text(tokens):
    """Convert tokens back to text."""
    return ENCODER.decode(tokens)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Token-based chunking with overlap."""
    tokens = tokenize_text(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = detokenize_text(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_category_file(file_path):
    """Process one JSON file and create chunks in the required format."""
    category_name = os.path.basename(file_path).replace(".json", "")
    output_file = os.path.join(OUTPUT_DIR, f"{category_name}_chunks.json")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks_list = []
    count = 1
    for item in tqdm(data, desc=f"Chunking {category_name}"):
        query_text = item.get("query-text", "").strip()
        responses = item.get("responses", [])
        title = item.get("query-title", "")
        citations = item.get("citations", [])
        url = item.get("query-url", "")
        # print(item)
        # break
        # Combine query + expert responses
        for i, resp in enumerate(responses, start=1):
            response = resp.get("response-text", "")
            combined_text = f"Expert {i}: {response}"
            
            for chunk in chunk_text(combined_text):
                chunks_list.append({
                    "content": chunk,
                    "metadata": {
                        "parent_id": count,
                        "title": title,
                        "query-text": query_text,
                        "citations": citations,
                        "url": url
                    }
                })
            count += 1

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(chunks_list, out, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(chunks_list)} chunks → {output_file}")

if __name__ == "__main__":
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            process_category_file(os.path.join(DATA_DIR, filename))

    print("\n🎯 All categories chunked and saved under:", OUTPUT_DIR)
