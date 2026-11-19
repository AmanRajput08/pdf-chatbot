import io
import hashlib
from typing import List, Dict, Any

import fitz
import faiss
import numpy as np
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

# -------------------- Configuration --------------------
OLLAMA_MODEL = "gemma3:1b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

st.set_page_config(
    page_title="Chat with your PDF | RAG",
    page_icon="📄",
    layout="wide"
)

# -------------------- Cached embedding model --------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    # Downloads the model on first run; later runs use the local cache.
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_pages(uploaded_files) -> List[Dict[str, Any]]:
    """Extract selectable text from each PDF page."""
    pages = []
    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "source": uploaded_file.name,
                    "page": page_number
                })
        doc.close()
    return pages


def split_into_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split page text into overlapping character chunks."""
    chunks = []
    for item in pages:
        text = " ".join(item["text"].split())
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": item["source"],
                    "page": item["page"]
                })
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks


def build_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    # Inner product on normalized vectors is cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve_chunks(question, index, chunks, embedding_model, top_k=TOP_K):
    query_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            result = dict(chunks[int(idx)])
            result["score"] = float(score)
            results.append(result)
    return results


def generate_answer(question, retrieved_chunks):
    context_parts = []
    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {item['source']}, page {item['page']}]\n{item['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant answering questions about the user's uploaded PDF documents.

Rules:
- Answer using only the CONTEXT below.
- If the answer is not present in the context, clearly say:
  "I couldn't find this information in the uploaded PDF."
- Do not invent facts.
- Keep the answer clear and concise.
- Mention the source filename and page number when useful.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return response["message"]["content"]


# -------------------- User Interface --------------------
st.title("📄 Chat with your PDF")
st.caption("A beginner-friendly Retrieval-Augmented Generation (RAG) project using FAISS and Ollama.")

with st.sidebar:
    st.header("Project details")
    st.write(f"**LLM:** `{OLLAMA_MODEL}`")
    st.write("**Embeddings:** `all-MiniLM-L6-v2`")
    st.write("**Vector search:** `FAISS (cosine similarity)`")
    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload PDF files\n"
        "2. Extract text with PyMuPDF\n"
        "3. Split text into chunks\n"
        "4. Create embeddings\n"
        "5. Search relevant chunks with FAISS\n"
        "6. Generate an answer with Gemma via Ollama"
    )
    if st.button("Clear session / start over"):
        for key in ["index", "chunks", "file_signature", "messages", "ready"]:
            st.session_state.pop(key, None)
        st.rerun()

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    signature_source = "|".join(
        f"{f.name}:{hashlib.md5(f.getvalue()).hexdigest()}" for f in uploaded_files
    )
    signature = hashlib.md5(signature_source.encode()).hexdigest()

    if st.session_state.get("file_signature") != signature:
        with st.spinner("Reading PDFs, creating chunks and building the vector index..."):
            pages = extract_pdf_pages(uploaded_files)
            chunks = split_into_chunks(pages)

            if chunks:
                embedding_model = load_embedding_model()
                index = build_faiss_index(chunks, embedding_model)
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["file_signature"] = signature
                st.session_state["messages"] = []
                st.session_state["ready"] = True
            else:
                st.session_state["ready"] = False
                st.error(
                    "No selectable text was found. This may be a scanned/image-only PDF. "
                    "This starter project does not include OCR."
                )

if st.session_state.get("ready"):
    st.success(
        f"Ready! Indexed {len(st.session_state['chunks'])} text chunks "
        "from your uploaded PDF file(s)."
    )

    # Display previous chat messages.
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                embedding_model = load_embedding_model()
                retrieved = retrieve_chunks(
                    question,
                    st.session_state["index"],
                    st.session_state["chunks"],
                    embedding_model
                )
                with st.spinner("Searching the PDF and generating an answer..."):
                    answer = generate_answer(question, retrieved)
                st.markdown(answer)

                with st.expander("See retrieved PDF passages and similarity scores"):
                    for item in retrieved:
                        st.markdown(
                            f"**{item['source']} — page {item['page']}** "
                            f"(similarity: {item['score']:.3f})"
                        )
                        st.write(item["text"])
                        st.divider()

                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                st.error(
                    "Something went wrong while calling the local model. "
                    "Check that Ollama is running and that `gemma3:1b` is installed."
                )
                st.code(str(error))
else:
    st.info("Upload a text-based PDF to begin. Your PDF text and chat are processed in this app.")
    st.markdown(
        "**Tip:** If a PDF is scanned (pages are just images), this starter version "
        "will need an OCR feature before it can read the text."
    )
