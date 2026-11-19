# RAG PDF Chatbot

A beginner-friendly PDF question-answering project using:

- **Streamlit** — web interface
- **PyMuPDF** — extract text from PDF pages
- **Sentence Transformers** — create text embeddings
- **FAISS** — similarity search over embeddings
- **Ollama + Gemma 3:1B** — generate answers locally

## 1. Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Ollama installed
- Gemma model downloaded in Ollama

Check the model:

```bash
ollama list
```

You should see `gemma3:1b`.

## 2. Open the project

Open this folder in VS Code and open its terminal.

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first installation can take time. Sentence Transformers downloads
`sentence-transformers/all-MiniLM-L6-v2` the first time it is loaded.
Keep an internet connection for this first download.

## 4. Make sure Ollama is running

In a terminal:

```bash
ollama run gemma3:1b
```

You can type a test question. After confirming it responds, type `/bye`.
Ollama normally keeps its local service available in the background.

If the app cannot connect, open the Ollama desktop application or start
the Ollama service, then try again.

## 5. Run the project

From the project folder, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will show a local address, usually:

```text
http://localhost:8501
```

Open that address in your browser, upload a PDF, and ask questions.

## 6. How the RAG pipeline works

1. Streamlit receives the uploaded PDF.
2. PyMuPDF extracts selectable text page by page.
3. Python splits the text into overlapping chunks.
4. Sentence Transformers converts each chunk into an embedding vector.
5. FAISS stores the vectors in a similarity-search index.
6. The user's question is converted into an embedding.
7. FAISS retrieves the most similar chunks.
8. The retrieved text is sent as context to Gemma 3:1B through Ollama.
9. Gemma generates an answer grounded in the retrieved passages.

## Important notes

- This starter project supports text-based PDFs. Scanned PDFs need OCR,
  which is not included yet.
- FAISS is a vector-search library/index, not a full database.
- The FAISS index and chat history are held in the Streamlit session in
  this starter version. Upload and process the PDF again in a new session.
- Do not upload confidential company documents unless you have permission.
  Ollama runs the LLM locally, but the embedding model may be downloaded
  from the internet on first use.
- Gemma 3:1B is a small model. Answers may be less capable than larger
  cloud models, especially for complex questions.
