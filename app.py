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
# 1️⃣ Page Setup
# ==========================================
st.set_page_config(page_title="Animal Recommendation System", layout="wide")

st.title("🐾 Animal Image Recommendation System")
st.write("Upload an animal image and get recommendations using:")
st.markdown("✅ KNN   ⚡ FAISS Flat   🚀 FAISS IVF")
st.divider()

# ==========================================
# 2️⃣ Dataset Setup (Download only once)
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
# 3️⃣ Load Image Paths
# ==========================================
image_paths = []

for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

if len(image_paths) == 0:
    st.error("❌ No images found in dataset folder!")
    st.stop()

st.success(f"✅ Total Dataset Images Found: {len(image_paths)}")

# ==========================================
# 4️⃣ Load Model (Cached)
# ==========================================
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

# ==========================================
# 5️⃣ Embedding Function
# ==========================================
def extract_embedding(img):
    img = img.resize((224, 224))
    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    emb = model.predict(img_array, verbose=0)
    return emb.flatten()

# ==========================================
# 6️⃣ Load / Create Embeddings (Cached)
# ==========================================
@st.cache_data(show_spinner=True)
def load_or_create_embeddings(image_paths, embedding_file):

    if os.path.exists(embedding_file):
        embeddings = np.load(embedding_file).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    st.warning("Creating embeddings for the first time...")

    embeddings = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            emb = extract_embedding(img)
            embeddings.append(emb)
        except:
            continue

    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)
    np.save(embedding_file, embeddings)

    return embeddings

embedding_file = "animal_embeddings.npy"
embeddings = load_or_create_embeddings(image_paths, embedding_file)

# Safety check
if len(embeddings) != len(image_paths):
    st.warning("⚠️ Embeddings mismatch. Rebuilding...")
    os.remove(embedding_file)
    st.rerun()

st.success("✅ Embeddings Ready")
st.write("Embedding Shape:", embeddings.shape)

# ==========================================
# 7️⃣ Load / Build FAISS Index (Cached)
# ==========================================
@st.cache_resource
def load_faiss_index(embeddings, dim):

    index_file = "faiss_index.bin"

    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        return index

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_file)
    return index

dim = embeddings.shape[1]
index_flat = load_faiss_index(embeddings, dim)

# ==========================================
# 8️⃣ KNN Model
# ==========================================
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(embeddings)

# ==========================================
# 9️⃣ FAISS IVF Model
# ==========================================
quantizer = faiss.IndexFlatIP(dim)
index_ivf = faiss.IndexIVFFlat(
    quantizer,
    dim,
    50,
    faiss.METRIC_INNER_PRODUCT
)

index_ivf.train(embeddings)
index_ivf.add(embeddings)
index_ivf.nprobe = 10

st.success("✅ Models Ready")
st.divider()

# ==========================================
# 🔟 Display Function
# ==========================================
def show_results(title, indices):
    st.subheader(title)
    cols = st.columns(5)
    for i, idx in enumerate(indices):
        try:
            img = Image.open(image_paths[idx])
            cols[i].image(img, width=150)
        except:
            continue

# ==========================================
# 1️⃣1️⃣ Upload Query Image
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

    faiss.normalize_L2(query_vector)

    k = 5

    # FAISS Flat
    D_flat, I_flat = index_flat.search(query_vector, k)
    best_similarity = D_flat[0][0]

    # Adaptive threshold
    dataset_mean_similarity = np.mean(
        index_flat.search(embeddings[:100], 2)[0][:, 1]
    )
    threshold = dataset_mean_similarity * 0.75

    st.write("Similarity Score:", round(float(best_similarity), 3))
    st.write("Threshold:", round(float(threshold), 3))

    if best_similarity < threshold:
        st.error("❌ No similar animal images found in dataset")
    else:
        show_results("⚡ FAISS Flat Recommendations", I_flat[0])

        distances_knn, indices_knn = knn.kneighbors(query_vector, k)
        show_results("✅ KNN Recommendations", indices_knn[0])

        D_ivf, I_ivf = index_ivf.search(query_vector, k)
        show_results("🚀 FAISS IVF Recommendations", I_ivf[0])
