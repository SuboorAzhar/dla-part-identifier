import os
import json
from pathlib import Path

# Configuration
DATA_DIR = Path("scraped_parts")
BASE_URL = "https://dla-part-images.s3.us-east-1.amazonaws.com/scraped_parts"
OUTPUT_FILE = "metadata_list.json"

# Rebuild metadata_list.json
metadata_list = []

for part_dir in sorted(DATA_DIR.glob("*_*_*")):
    if not part_dir.is_dir():
        continue

    meta_path = part_dir / "metadata.json"
    if not meta_path.exists():
        continue

    with open(meta_path, "r") as f:
        meta = json.load(f)

    folder_name = part_dir.name
    cover_url = f"{BASE_URL}/{folder_name}/cover.png"

    support_urls = []
    for img_file in sorted(part_dir.glob("support_*.png")):
        support_urls.append(f"{BASE_URL}/{folder_name}/{img_file.name}")

    metadata_list.append({
        "meta": meta,
        "folder": folder_name,
        "cover_path": cover_url,
        "support_paths": support_urls
    })

# Write final metadata list
with open(OUTPUT_FILE, "w") as f:
    json.dump(metadata_list, f, indent=2)

print(f"✅ Updated metadata written to {OUTPUT_FILE}")
