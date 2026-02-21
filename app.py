import streamlit as st
import numpy as np
import os
from PIL import Image
import zipfile

import faiss
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1️⃣ Streamlit Page Setup
# ==========================================
st.set_page_config(page_title="Animal Recommendation System", layout="wide")

st.title("🐾 Animal Image Recommendation System")
st.write("Upload an animal image and get recommendations using:")
st.markdown("✅ KNN   ⚡ FAISS Flat   🚀 FAISS IVF")
st.divider()

# ==========================================
# 2️⃣ Kaggle Dataset Auto Download
# ==========================================
DATASET_FOLDER = "animals"

if not os.path.exists(DATASET_FOLDER):

    st.warning("Dataset not found. Downloading from Kaggle...")

    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

    os.system(
        "kaggle datasets download -d iamsouravbanerjee/animal-image-dataset-90-different-animals"
    )

    with zipfile.ZipFile(
        "animal-image-dataset-90-different-animals.zip", "r"
    ) as zip_ref:
        zip_ref.extractall("dataset")

    os.rename("dataset/animals", "animals")

    st.success("✅ Dataset Downloaded Successfully!")

# ==========================================
# 3️⃣ Load Dataset Image Paths
# ==========================================
image_paths = []

for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

st.success(f"✅ Total Dataset Images Found: {len(image_paths)}")

if len(image_paths) == 0:
    st.error("No images found in dataset folder!")
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

    img = img.resize((224, 224))
    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    emb = model.predict(img_array, verbose=0)
    return emb.flatten()

# ==========================================
# 6️⃣ Load or Create Embeddings (Normalized)
# ==========================================
embedding_file = "animal_embeddings.npy"

if not os.path.exists(embedding_file):

    st.warning("Embeddings not found. Creating embeddings...")

    embeddings = []
    progress = st.progress(0)

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            emb = extract_embedding(img)
            embeddings.append(emb)
        except:
            continue

        progress.progress((i + 1) / len(image_paths))

    embeddings = np.array(embeddings).astype("float32")

    # Normalize embeddings (CRITICAL)
    faiss.normalize_L2(embeddings)

    np.save(embedding_file, embeddings)

    st.success("✅ Embeddings Extracted and Saved!")

else:
    embeddings = np.load(embedding_file).astype("float32")
    faiss.normalize_L2(embeddings)
    st.success("✅ Embeddings Loaded Successfully!")

st.write("Embeddings Shape:", embeddings.shape)

# ==========================================
# 7️⃣ Build Recommendation Models (Cosine)
# ==========================================

# ---- KNN (cosine distance) ----
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(embeddings)

# ---- FAISS Flat (Inner Product = Cosine) ----
dim = embeddings.shape[1]
index_flat = faiss.IndexFlatIP(dim)
index_flat.add(embeddings)

# ---- FAISS IVF (Inner Product) ----
nlist = 50
quantizer = faiss.IndexFlatIP(dim)

index_ivf = faiss.IndexIVFFlat(
    quantizer,
    dim,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

index_ivf.train(embeddings)
index_ivf.add(embeddings)
index_ivf.nprobe = 10

st.success("✅ All Recommendation Models Ready!")
st.divider()

# ==========================================
# 8️⃣ Display Recommendation Function
# ==========================================
def show_results(title, indices):
    st.subheader(title)
    cols = st.columns(5)
    for i, idx in enumerate(indices):
        img = Image.open(image_paths[idx])
        cols[i].image(img, width=150)

# ==========================================
# 9️⃣ Upload Query Image
# ==========================================
uploaded_file = st.file_uploader(
    "📌 Upload Query Animal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="📷 Query Image", width=250)

    query_vector = extract_embedding(query_img).astype("float32")
    query_vector = np.expand_dims(query_vector, axis=0)

    # Normalize query embedding
    faiss.normalize_L2(query_vector)

    k = 5
    threshold = 0.65   # Adjust if needed

    # ==============================
    # KNN Recommendations
    # ==============================
    distances_knn, indices_knn = knn.kneighbors(query_vector, k)
    best_similarity_knn = 1 - distances_knn[0][0]

    if best_similarity_knn < threshold:
        st.error("❌ No similar animal images found (KNN)")
    else:
        show_results("✅ KNN Recommendations", indices_knn[0])

    # ==============================
    # FAISS Flat Recommendations
    # ==============================
    D_flat, I_flat = index_flat.search(query_vector, k)
    best_similarity_flat = D_flat[0][0]

    if best_similarity_flat < threshold:
        st.error("❌ No similar animal images found (FAISS Flat)")
    else:
        show_results("⚡ FAISS Flat Recommendations", I_flat[0])

    # ==============================
    # FAISS IVF Recommendations
    # ==============================
    D_ivf, I_ivf = index_ivf.search(query_vector, k)
    best_similarity_ivf = D_ivf[0][0]

    if best_similarity_ivf < threshold:
        st.error("❌ No similar animal images found (FAISS IVF)")
    else:
        show_results("🚀 FAISS IVF Recommendations", I_ivf[0])
