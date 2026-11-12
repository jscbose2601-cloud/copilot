import os
import streamlit as st
import ollama
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

st.set_page_config(
    page_title="Enterprise AI Copilot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedding_model()

st.title("🤖 Enterprise Knowledge Copilot")
st.caption("Your intelligent company document assistant")

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("📁 Knowledge Center")

    uploaded_files = st.file_uploader(
        "Upload company PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        all_chunks = []

        if not uploaded_files:
            st.warning("Upload PDF documents first.")

        else:
            with st.spinner("Processing documents..."):

                for file in uploaded_files:
                    reader = PdfReader(file)

                    for page_num, page in enumerate(reader.pages):
                        content = page.extract_text() or ""

                        words = content.split()

                        for i in range(0, len(words), 120):
                            chunk = " ".join(
                                words[i:i + 120]
                            )

                            if chunk.strip():
                                all_chunks.append({
                                    "text": chunk,
                                    "source": file.name,
                                    "page": page_num + 1
                                })

                if all_chunks:
                    texts = [
                        item["text"] for item in all_chunks
                    ]

                    embeddings = embedder.encode(
                        texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )

                    embeddings = np.asarray(
                        embeddings, dtype="float32"
                    )

                    index = faiss.IndexFlatIP(
                        embeddings.shape[1]
                    )

                    index.add(embeddings)

                    st.session_state.chunks = all_chunks
                    st.session_state.index = index

                    st.success(
                        f"{len(all_chunks)} chunks indexed!"
                    )
                else:
                    st.warning("No readable PDF text found.")

    st.divider()

    st.write("📚 Indexed chunks:",
             len(st.session_state.chunks))

    if st.button("Clear Chat"):
        st.session_state.history = []

def retrieve_documents(question, k=3):

    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding, dtype="float32"
    )

    scores, ids = st.session_state.index.search(
        query_embedding,
        min(k, len(st.session_state.chunks))
    )

    results = []

    for idx in ids[0]:
        if idx >= 0:
            results.append(
                st.session_state.chunks[idx]
            )

    return results

def generate_answer(question):

    documents = retrieve_documents(question)

    context = "\n\n".join([
        f"Source: {doc['source']}, "
        f"Page: {doc['page']}\n{doc['text']}"
        for doc in documents
    ])

    prompt = f"""
You are an Enterprise Knowledge Assistant.

Answer using ONLY the provided context.

If the answer is not present, say:
"I could not find this information
in the uploaded documents."

Do not invent company policies.

CONTEXT:
{context}

QUESTION:
{question}

Give a clear and simple answer.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"], documents

# Main chat area

st.subheader("💬 Ask your company documents")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask something about your documents..."
)

if question:

    if st.session_state.index is None:
        st.warning("Please upload and process PDFs first.")

    else:
        st.session_state.history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("AI is thinking..."):

                try:
                    answer, sources = generate_answer(
                        question
                    )

                    st.markdown(answer)

                    st.markdown("### 📚 Sources")

                    for doc in sources:
                        st.caption(
                            f"📄 {doc['source']} | "
                            f"Page {doc['page']}"
                        )

                    st.session_state.history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
