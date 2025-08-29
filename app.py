import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials


# ========== SETTINGS ==========
st.set_page_config(page_title="StudyMate – PDF Q&A Assistant", page_icon="📘", layout="wide")

# Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# --- Text chunking
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [
        " ".join(words[i: i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# --- Create embeddings
def embed_chunks(chunks):
    clean_chunks = [str(c).strip() for c in chunks if isinstance(c, str) and str(c).strip() != ""]
    if not clean_chunks:
        return np.array([]), []
    embeddings = embedder.encode(clean_chunks, convert_to_numpy=True)
    return embeddings, clean_chunks

# --- Retrieve top chunks
def retrieve_chunks(question, embeddings, chunks, top_n=3):
    q_emb = embedder.encode(question)
    sims = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    top_idx = np.argsort(sims)[-top_n:][::-1]
    return [chunks[i] for i in top_idx]

# --- Ask IBM WatsonX
def ask_ibm(question, context):
    cred = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key="_qvjjOY6Ia1Fz65hPmH3m28HaYhplEZahB1uZAJ3G5zS",   # 🔑 Replace with your IBM API Key
    )
    model = Model(
        model_id="ibm/granite-3-2b-instruct",
        credentials=cred,
        project_id="427c361c-abeb-43e0-a72c-de9da5f61197", # 🔑 Replace with your IBM Project ID
    )
    
    prompt = f"""
    You are an AI assistant answering based only on the given context.
    Write a detailed, structured answer with **at least 120 words**.
    Avoid short replies. Provide explanations, step-by-step reasoning, and examples.

    Context:
    {context}

    Question: {question}
    """

    params = {
        "decoding_method": "greedy",
        "max_new_tokens": 300,   # 🔑 allow enough space for long answers
        "min_new_tokens": 120,   # 🔑 force at least ~120 words
        "temperature": 0.7
    }

    return model.generate_text(prompt=prompt, params=params)


# ========== UI ==========
st.title("📘 StudyMate – PDF Q&A Assistant")
st.markdown("Upload a PDF and ask questions – AI will answer using **WatsonX + embeddings** 🚀")

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

# Process PDF
if uploaded_file:
    reader = PdfReader(uploaded_file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.extend(chunk_text(text))

    st.session_state.embeddings, st.session_state.chunks = embed_chunks(chunks)

    if len(st.session_state.chunks) == 0:
        st.error("❌ No readable text found in PDF. Try a different file.")
    else:
        st.success(f"✅ Loaded {len(st.session_state.chunks)} text chunks from PDF.")

        # Question box
        st.subheader("❓ Ask a Question")
        question = st.text_input("Type your question here...")

        if question:
            if "embeddings" in st.session_state and len(st.session_state.embeddings) > 0:
                rel_chunks = retrieve_chunks(question, st.session_state.embeddings, st.session_state.chunks)
                context = "\n\n".join(rel_chunks)
                with st.spinner("🤔 Thinking..."):
                    answer = ask_ibm(question, context)

                # Display results
                st.markdown("### 📌 Answer")
                st.info(answer)

                # Show retrieved context
                with st.expander("📑 View supporting context"):
                    st.write(context)
            else:
                st.warning("⚠️ Upload a PDF with readable text first.")
else:
    st.info("📤 Please upload a PDF from the sidebar to begin.")
