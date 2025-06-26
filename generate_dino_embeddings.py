import os
import json
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms

# --------------------
# Config
# --------------------
IMG_DIM = 518
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/dinov2-large"
DATA_DIR = Path("scraped_parts")
EMBED_FILE = "embeddings.npy"
META_FILE = "metadata_list.json"

# --------------------
# Load DINOv2
# --------------------
print(f"🔍 Loading DINOv2 model: {MODEL_NAME}")
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

transform = transforms.Compose([
    transforms.Resize((IMG_DIM, IMG_DIM)),
    transforms.CenterCrop((IMG_DIM, IMG_DIM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

# --------------------
# Embedding function
# --------------------
def encode_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    except (UnidentifiedImageError, OSError) as e:
        print(f"❌ Skipping unreadable image {img_path}: {e}")
        return None
    except Exception as e:
        print(f"❌ Failed on {img_path}: {e}")
        return None

# --------------------
# Walk folders & encode (average all .png images)
# --------------------
embeddings = []
metadata = []

print("📁 Scanning folders...")
all_parts = sorted(DATA_DIR.glob("*_*_*"))

for idx, part_dir in enumerate(tqdm(all_parts, desc="🧠 Embedding Parts")):
    if not part_dir.is_dir():
        continue
    meta_path = part_dir / "metadata.json"
    if not meta_path.exists():
        continue

    emb_list = []
    images = sorted(part_dir.glob("*.png"))
    print(f"\n📦 Processing {part_dir.name} ({len(images)} images)...")

    for i, img_path in enumerate(images):
        print(f"  ➜ [{i+1}/{len(images)}] {img_path.name}")
        emb = encode_image(img_path)
        if emb is not None:
            emb_list.append(emb)

    if not emb_list:
        continue

    final_emb = np.mean(emb_list, axis=0)

    with open(meta_path) as f:
        meta_json = json.load(f)

    metadata.append({
        "meta": meta_json,
        "folder": str(part_dir),
        "cover_path": str(part_dir / "cover.png")
    })
    embeddings.append(final_emb)

print(f"✅ Encoded {len(embeddings)} parts using average of all images")

np.save(EMBED_FILE, np.array(embeddings))
with open(META_FILE, "w") as f:
    json.dump(metadata, f, indent=2)

print("💾 Saved embeddings and metadata.")
