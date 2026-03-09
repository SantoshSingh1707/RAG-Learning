# AI-Powered Conversational RAG System

A sophisticated **Retrieval Augmented Generation (RAG)** system that combines semantic search with LLM intelligence to provide precise answers from your documents. 

Built with **Streamlit**, **LangChain**, **ChromaDB**, and **Mistral AI**, this application supports PDF/TXT ingestion, OCR for image-based documents, and interactive chat with source citations.

## 📋 Overview

This project provides a professional-grade RAG workflow:
- **Dynamic Ingestion**: Upload documents via a web interface or bulk process from local directories.
- **OCR Support**: Automatically detects and extracts text from image-based or scanned PDFs using EasyOCR.
- **Intelligent Chunking**: Sophisticated text splitting with recursive strategies to maintain semantic context.
- **Vector Storage**: Persistent storage of embeddings in ChromaDB for fast retrieval.
- **Conversational AI**: Integrates Mistral AI (`mistral-small-2506`) for natural, context-aware responses.
- **Interactive UI**: A full-featured Streamlit dashboard for querying, document management, and source exploration.

## 🚀 Key Features

✅ **Interactive Chat Interface**: Conversational UI with persistent chat memory.  
✅ **Multi-format Support**: Process PDFs, text files, and scanned documents.  
✅ **OCR Fallback**: Automated extraction for non-searchable PDFs.  
✅ **Source Citations**: Every answer includes clickable source links with similarity scores.  
✅ **Document Management**: Upload new files or delete existing ones directly from the UI.  
✅ **Response Export**: Download generated answers and metadata as text files.  
✅ **GPU Acceleration**: CUDA support for high-speed embedding generation.  
✅ **Source Filtering**: Narrow your search to specific documents in your database.

## 📦 Core Stack

- **Frontend**: `Streamlit`
- **Orchestration**: `LangChain` & `LangChain-Community`
- **LLM**: `Mistral AI` (via `langchain-mistralai`)
- **Embeddings**: `Sentence-Transformers` (`multi-qa-MiniLM-L6-cos-v1`)
- **Database**: `ChromaDB` (Vector Store)
- **OCR**: `EasyOCR` & `PyMuPDF` (Fitz)
- **Environment**: `Python 3.11+`, `uv` for package management

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SantoshSingh1707/RAG-Learning.git
   cd RAG-Learning
   ```

2. **Setup Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   MISTRAL_API_KEY=your_api_key_here
   ```

3. **Install Dependencies**
   Using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU support, ensure CUDA-compatible torch is installed (see `pyproject.toml` for details).*

## 📂 Project Structure

```
RAG-Learning/
├── app.py                  # Main Streamlit application
├── ingest_data.py          # CLI tool for bulk document ingestion
├── requirements.txt        # Production dependencies
├── pyproject.toml         # Project configuration & UV sync
├── .env                   # API keys and environment configs
├── src/                    # Core source code
│   ├── data_loader.py      # PDF/TXT/OCR processing logic
│   ├── embedding.py        # Embedding generation manager
│   ├── vector_store.py     # ChromaDB interface
│   └── search.py           # RAG retrieval and LLM integration
├── data/
│   ├── pdf/               # Local PDF storage
│   ├── textfiles/         # Local text file storage
│   └── vector_store/      # ChromaDB persistent database
└── notebook/               # Experimental Jupyter notebooks
```

## 🔧 Usage

### 💻 Web Application (Recommended)
Launch the interactive dashboard to upload, manage, and chat with your documents:
```bash
streamlit run app.py
```

### 🛠️ CLI Bulk Ingestion
To populate your vector database with existing files in the `data/` folder:
```bash
python ingest_data.py
```

## 🏗️ Technical Implementation

### 1. Document Processing & OCR
The system uses a hierarchical approach:
1. Attempt text extraction via `PyPDFLoader`.
2. If no text is found (scanned document), fall back to `EasyOCR` for page-by-page vision processing.
3. Metadata like `source_file`, `page`, and `file_type` are preserved for citation.

### 2. Semantic Retrieval
Queries are embedded using the `multi-qa-MiniLM-L6-cos-v1` model, optimized for Q&A tasks. Retrieval uses **cosine similarity** conversion:
```python
similarity_score = 1 / (1 + distance)
```

### 3. LLM Enhancement
The `rag_enhanced` function combines retrieved context with user queries, sending a structured prompt to Mistral AI. The system maintains conversation history for coherent multi-turn dialogues.

## 📝 Configuration

Key parameters adjustable via the UI sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Top-K Documents | 5 | Number of relevant chunks retrieved |
| Similarity Threshold | 0.35 | Minimum relevance score (0-1) |
| Model | mistral-small | LLM used for answer generation |
| Chunk Size | 1000 | Characters per segment during ingestion |

## 🚀 Future Enhancements

- [ ] Support for Excel (`.xlsx`) and Word (`.docx`) documents.
- [ ] Integration with more LLM providers (OpenAI, Anthropic).
- [ ] Advanced "Re-ranking" (Cross-Encoders) for higher precision.
- [ ] User authentication and private workspaces.

## 📄 License & Credits

Created by **Santosh**  
Updated: March 9, 2026

*Educational project demonstrating modern RAG architectures.*
