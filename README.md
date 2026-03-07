# RAG Pipeline - Retrieval Augmented Generation

A Python-based **Retrieval Augmented Generation (RAG)** pipeline that processes PDF documents, creates semantic embeddings, and enables intelligent document retrieval using similarity search.

## 📋 Overview

This project demonstrates a complete RAG workflow:
- **Data Ingestion**: Load PDF and text documents from specified directories
- **Text Processing**: Split documents into manageable chunks with configurable overlap
- **Embedding Generation**: Convert text chunks into semantic embeddings using SentenceTransformer
- **Vector Storage**: Store embeddings in a persistent ChromaDB vector database
- **Intelligent Retrieval**: Retrieve relevant documents based on semantic similarity to user queries

## 🚀 Features

✅ Multi-format document loading (PDF, TXT)  
✅ Intelligent text chunking with recursive splitting  
✅ Semantic embeddings using `sentence-transformers`  
✅ Persistent vector database with ChromaDB  
✅ Similarity-based document retrieval  
✅ Metadata preservation throughout the pipeline  
✅ Configurable chunk size, overlap, and retrieval thresholds  

## 📦 Dependencies

- `langchain` - Document processing and utilities
- `langchain-core` - Core LangChain components
- `langchain-community` - Community integrations (PDF loaders, etc.)
- `langchain-text-splitters` - Text splitting utilities
- `sentence-transformers` - Semantic embeddings (all-MiniLM-L6-v2)
- `chromadb` - Vector database
- `pypdf` & `pymupdf` - PDF document processing
- `scikit-learn` - ML utilities
- `numpy` - Numerical computing

## 📥 Installation

1. **Clone or download the project**
   ```bash
   cd RAG-Learning
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure

```
RAG-Learning/
├── main.py                    # Main entry point
├── requirements.txt           # Project dependencies
├── pyproject.toml            # Project configuration
├── README.md                 # This file
├── .env                      # Environment configuration
├── data/
│   ├── pdf/                  # PDF documents to process
│   ├── textfiles/            # Text documents
│   │   ├── python_intro.txt
│   │   └── machine_learning.txt
│   └── vector_store/         # ChromaDB storage
│       └── chroma.sqlite3
└── notebook/
    ├── document.ipynb        # Data ingestion demo
    └── pdf_loader.ipynb      # Complete RAG pipeline
```

## 🔧 Usage

### Option 1: Using Jupyter Notebooks

**Data Ingestion Demo:**
```bash
jupyter notebook notebook/document.ipynb
```

**Complete RAG Pipeline:**
```bash
jupyter notebook notebook/pdf_loader.ipynb
```

### Option 2: Run from Command Line

```bash
python main.py
```

## 🏗️ Pipeline Components

### 1. Document Loading
```python
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

# Load all PDFs from directory
loader = DirectoryLoader("data/pdf", 
                        glob="**/*.pdf",
                        loader_cls=PyMuPDFLoader)
documents = loader.load()
```

### 2. Text Splitting
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
```

### 3. Embedding Generation
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
```

### 4. Vector Storage
```python
import chromadb

client = chromadb.PersistentClient(path="data/vector_store")
collection = client.get_or_create_collection("pdf_documents")
collection.add(ids=ids, embeddings=embeddings, documents=texts)
```

### 5. Document Retrieval
```python
# Query similar documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# Results include: ids, documents, metadatas, distances
```

## 📊 Key Classes

### `EmbeddingManager`
Handles embedding generation using SentenceTransformer models.
- `generate_embeddings(texts)` - Convert text to embeddings

### `VectorStore`
Manages ChromaDB vector database operations.
- `initialize_store()` - Setup persistent storage
- `add_documents(documents, embeddings)` - Store document embeddings

### `RAGRetrival`
Performs semantic similarity search and document retrieval.
- `retrive(query, top_k=5, score_threshold=0.3)` - Find relevant documents

## 🔍 Retrieval Details

The retrieval system uses **cosine distance** with similarity conversion:

```python
similarity_score = 1 / (1 + distance)
```

This formula converts distances to a 0-1 similarity range:
- Default threshold: **0.3** (adjustable)
- Default top_k: **5** results
- Returns: Document content, metadata, similarity score, and rank

## 🎯 Example Query

```python
# Initialize retriever
rag_retriever = RAGRetrival(vectorstore, embedding_manager)

# Retrieve relevant documents
results = rag_retriever.retrive(
    query="What is machine learning?",
    top_k=5,
    score_threshold=0.3
)

# Results contain:
# - id: Document identifier
# - content: Document text
# - metadata: Source info, page numbers, etc.
# - similarity_score: Relevance score (0-1)
# - distance: Raw distance metric
# - rank: Retrieval rank
```

## 📝 Configuration

Key parameters you can adjust:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Embedding Model | `EmbeddingManager.__init__()` | all-MiniLM-L6-v2 | SentenceTransformer model |
| Chunk Size | `split_document()` | 1000 | Characters per chunk |
| Chunk Overlap | `split_document()` | 200 | Overlapping characters |
| Similarity Threshold | `RAGRetrival.retrive()` | 0.3 | Minimum relevance score |
| Top-K Results | `RAGRetrival.retrive()` | 5 | Number of results to return |
| Persist Directory | `VectorStore.__init__()` | ../data/vector_store | Vector DB location |

## 📚 Data Format

### Input Documents
- **PDFs**: `data/pdf/*.pdf`
- **Text Files**: `data/textfiles/*.txt`

### Document Metadata
Each document preserves:
- `source`: Original file name
- `source_file`: PDF file name
- `file_type`: Document type (pdf/txt)
- `page`: Page number (for PDFs)
- `doc_index`: Chunk index
- `context_length`: Character count

## 🐛 Troubleshooting

### Issue: Retrieval returns 0 documents
- **Cause**: Similarity threshold too high
- **Solution**: Lower `score_threshold` parameter (try 0.2-0.4)

### Issue: Slow embedding generation
- **Cause**: Large number of documents or chunks
- **Solution**: Adjust `chunk_size` parameter or use GPU (see sentence-transformers docs)

### Issue: OutOfMemory errors
- **Cause**: Processing too many large PDFs
- **Solution**: Process PDFs in batches or increase chunk size

## 🚀 Future Enhancements

- [ ] Multi-GPU support for faster embeddings
- [ ] Support for other embedding models (OpenAI, Cohere)
- [ ] Integration with LLMs for answer generation
- [ ] Web UI for document querying
- [ ] Batch processing optimization
- [ ] Document metadata filtering

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Santosh**  
Created: March 7, 2026

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📞 Support

For issues or questions, please check the Jupyter notebooks in the `notebook/` directory for detailed examples and usage patterns.

---

**Happy RAG-ing! 🚀**
