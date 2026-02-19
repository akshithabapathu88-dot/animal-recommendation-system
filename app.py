import streamlit as st
import numpy as np
import os
from PIL import Image

import faiss
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1️⃣ Streamlit Page Setup
# ==========================================
st.set_page_config(page_title="Animal Recommendation System", layout="wide")

st.title("🐾 Animal Image Recommendation System")
st.write("Upload an animal image and get similar recommendations using:")
st.markdown("✅ KNN   ⚡ FAISS Flat   🚀 FAISS IVF")

st.divider()


# ==========================================
# 2️⃣ Dataset Folder Path
# ==========================================
dataset_path = "animals"   # Folder must exist

if not os.path.exists(dataset_path):
    st.error("❌ Dataset folder not found! Please create an 'animals/' folder.")
    st.stop()


# ==========================================
# 3️⃣ Load Dataset Image Paths
# ==========================================
image_paths = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

st.success(f"✅ Total Dataset Images Found: {len(image_paths)}")

# Stop if dataset is empty
if len(image_paths) == 0:
    st.error("❌ No images found inside 'animals/' folder!")
    st.info("👉 Please add images like cat.jpg, dog.jpg, tiger.jpg inside animals/")
    st.stop()


# ==========================================
# 4️⃣ Load MobileNetV2 Model
# ==========================================
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

model = load_model()


# ==========================================
# 5️⃣ Extract Embedding Function
# ==========================================
def extract_embedding(img):

    img = img.convert("RGB")  # Ensure 3 channels
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    emb = model.predict(img_array, verbose=0)
    return emb.flatten()


# ==========================================
# 6️⃣ Load or Create Embeddings
# ==========================================
embedding_file = "animal_embeddings.npy"

# Always regenerate if file is too small
if os.path.exists(embedding_file) and os.path.getsize(embedding_file) < 5000:
    os.remove(embedding_file)
    st.warning("⚠️ Old embedding file was empty. Deleted it!")

if not os.path.exists(embedding_file):

    st.warning("⚠️ Embedding file not found!")
    st.write("Creating embeddings now... (first run may take few minutes)")

    embeddings = []
    progress = st.progress(0)

    for i, img_path in enumerate(image_paths):

        try:
            img = Image.open(img_path)
            emb = extract_embedding(img)
            embeddings.append(emb)

        except Exception as e:
            st.error(f"Error processing image: {img_path}")
            st.write(e)

        progress.progress((i + 1) / len(image_paths))

    embeddings = np.array(embeddings)

    # Save embeddings
    np.save(embedding_file, embeddings)

    st.success("✅ Embeddings Extracted and Saved!")

else:
    embeddings = np.load(embedding_file)
    st.success("✅ Embeddings Loaded Successfully!")

st.write("📌 Embeddings Shape:", embeddings.shape)


# ==========================================
# 7️⃣ Build Recommendation Models
# ==========================================

# ---- KNN ----
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(embeddings)

# ---- FAISS Flat ----
dim = embeddings.shape[1]
index_flat = faiss.IndexFlatL2(dim)
index_flat.add(embeddings)

# ---- FAISS IVF ----
nlist = 50
quantizer = faiss.IndexFlatL2(dim)

index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
index_ivf.train(embeddings)
index_ivf.add(embeddings)

index_ivf.nprobe = 10

st.success("✅ Recommendation Models Ready!")

st.divider()


# ==========================================
# 8️⃣ Display Recommendation Function
# ==========================================
def show_results(title, indices):

    st.subheader(title)

    cols = st.columns(5)

    for i, idx in enumerate(indices):

        if idx < len(image_paths):
            img = Image.open(image_paths[idx])
            cols[i].image(img, width=150)


# ==========================================
# 9️⃣ Upload Query Image UI
# ==========================================
uploaded_file = st.file_uploader(
    "📌 Upload Query Animal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load Uploaded Image
    query_img = Image.open(uploaded_file)

    st.image(query_img, caption="📷 Query Image", width=250)

    # Extract Query Embedding
    query_vector = extract_embedding(query_img)

    k = 5

    # ==============================
    # KNN Search
    # ==============================
    distances_knn, indices_knn = knn.kneighbors([query_vector], k)

    show_results("✅ KNN Recommendations", indices_knn[0])

    # ==============================
    # FAISS Flat Search
    # ==============================
    D_flat, I_flat = index_flat.search(np.array([query_vector]), k)

    show_results("⚡ FAISS Flat Index Recommendations", I_flat[0])

    # ==============================
    # FAISS IVF Search
    # ==============================
    D_ivf, I_ivf = index_ivf.search(np.array([query_vector]), k)

    show_results("🚀 FAISS IVF Index Recommendations", I_ivf[0])
