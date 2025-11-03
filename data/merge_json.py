import json

# List of your input JSON files
files = ["train.json", "test.json", "dev.json"]

merged_data = []

# Read and merge all files
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Ensure the file contains a list
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)

# Write merged content to a new file
with open("queries_data.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"✅ All files merged into queries_data.json with {len(merged_data)} entries")
