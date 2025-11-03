import json
import os
from collections import defaultdict

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "queries_data.json"          # your master JSON file
OUTPUT_DIR = "./grouped_categories" # folder to save grouped files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------
# 2️⃣ Group queries by category
# -------------------------------
category_groups = defaultdict(list)

for entry in data:
    category = entry.get("query-category", "Unknown").strip()
    category_groups[category].append(entry)

# -------------------------------
# 3️⃣ Save each category to its own file
# -------------------------------
for category, items in category_groups.items():
    filename = category.replace(" ", "_").lower() + ".json"
    path = os.path.join(OUTPUT_DIR, filename)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Saved {len(items)} queries → {filename}")

print("\n🎯 Grouping complete!")
print(f"All files saved under: {OUTPUT_DIR}")
