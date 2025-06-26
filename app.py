import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import requests
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
import torch

# -------------------------
# Load Data
# -------------------------
if not os.path.exists("embeddings.npy") or not os.path.exists("metadata_list.json"):
    st.error("❌ embeddings.npy or metadata_list.json not found. Please run preprocessing scripts.")
    st.stop()

embeddings = np.load("embeddings.npy")

with open("metadata_list.json") as f:
    metadata = json.load(f)

# -------------------------
# UI Functions
# -------------------------
def render_part_details(meta):
    st.markdown("### 🧾 Part Details")
    st.write(f"**NSN:** {meta.get('NSN', 'N/A')}")
    st.write(f"**Part Number:** {meta.get('Part Number', 'N/A')}")
    st.write(f"**CAGE:** {meta.get('CAGE', 'N/A')}")
    st.write(f"**Name:** {meta.get('Name', 'N/A')}")
    st.markdown(f"[🔗 Product Page]({meta.get('Product URL', '#')})")

def render_image_grid(cover_url, support_urls):
    st.markdown("### 📦 Part Images")
    cols = st.columns(3)

    with cols[0]:
        st.image(cover_url, caption="Cover", use_column_width=True)

    for i, url in enumerate(support_urls):
        with cols[(i + 1) % 3]:
            st.image(url, caption=f"Support {i+1}", use_column_width=True)

# -------------------------
# Model Setup
# -------------------------
@st.cache_resource
def load_model():
    model_name = "facebook/dinov2-large"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, processor

model, processor = load_model()

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.CenterCrop((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

# -------------------------
# App Layout
# -------------------------
st.set_page_config(layout="wide")
st.title("🔍 DLA Part Identifier (Image + Text Search)")

mode = st.radio("Choose Mode", ["Search by Part Number / NSN", "Search by Image"], horizontal=True)

if mode == "Search by Part Number / NSN":
    query = st.text_input("Enter NSN or Part Number")
    if query:
        query = query.strip().lower()
        found = False
        for part in metadata:
            meta = part["meta"]
            if query in meta.get("NSN", "").lower() or query in meta.get("Part Number", "").lower():
                render_part_details(meta)
                render_image_grid(part["cover_url"], part["support_urls"])
                found = True
                break
        if not found:
            st.warning("❌ No matching part found.")

else:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        sims = np.dot(embeddings, emb)
        idx = int(np.argmax(sims))

        best_match = metadata[idx]
        st.success("✅ Closest Match Found:")
        render_part_details(best_match["meta"])
        render_image_grid(best_match["cover_url"], best_match["support_urls"])
