"""
RAG Pipeline - Complete implementation for PDF processing, embeddings, vector storage, and LLM integration
"""

import os
import numpy as np
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
except ImportError:
    from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding and Vector DB
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity

# LLM
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()


# ==================== PDF Processing ====================

def process_all_pdf(pdf_directory):
    """Load all PDF files from a directory recursively"""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} pdf files to process")

    for pdf_file in pdf_files:
        print(f"processing : {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")
        
        except Exception as e:
            print(f"Error : {e}")
        
        print(f"Total documents loaded : {len(all_documents)}")
    return all_documents


def split_document(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks with overlap"""
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_spliter.split_documents(documents)
    print(f"split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print("Example Chunk: ")
        print(f"Content : {split_docs[0].page_content[:200]}")
        print(f"Metadata : {split_docs[0].metadata}")
    return split_docs


# ==================== Embedding Manager ====================

class EmbeddingManager:
    """Manages sentence embeddings using SentenceTransformer"""
    
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            print(f"Loading embedding model : {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded successfully. Embedding dimension : {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model : {self.model_name}")
            raise e
    
    def generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings with optional query prefix"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Add prefix for better Q&A embeddings
        if is_query:
            texts = [f"query: {text}" for text in texts]
        else:
            texts = [f"passage: {text}" for text in texts]
        
        print(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings


# ==================== Vector Store ====================

class VectorStore:
    """ChromaDB Vector Store for document embeddings"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.initialize_store()
    
    def initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection : {self.collection_name}")
            print(f"Existing documents in collection : {self.collection.count()}")
        except Exception as e:
            print(f"Error while initializing vector store {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and embeddings to vector store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['context_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"Successfully added {len(documents)} documents in vector store")
            print(f"Total Documents in collection : {self.collection.count()}")
        
        except Exception as e:
            print(f"Error adding documents to vector store : {e}")
            raise 


# ==================== RAG Retrieval ====================

class RAGRetrieval:
    """Retrieves documents from vector store based on query similarity"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar documents"""
        print(f"Retrieving documents for query : '{query}' ")
        print(f"Top_k : {top_k} , Score_threshold : {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query], is_query=True)[0]
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 / (1 + distance)
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval {e}")
            return []


# ==================== RAG Pipelines ====================

def rag_simple(query, retriever, llm, top_k=3):
    """Simple RAG pipeline - retrieves context and generates answer"""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found"
    
    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke([prompt])
    return response.content


def rag_enhanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    """Enhanced RAG pipeline with sources and confidence"""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    
    if not results:
        return {
            'answer': 'No relevant context found',
            'sources': [],
            'confidence': 0.0
        }
    
    context = "\n\n".join([doc['content'] for doc in results])
    
    sources = [{
        'source_file': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'similarity_score': doc['similarity_score'],
        'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content']
    } for doc in results]
    
    confidence = max([doc['similarity_score'] for doc in results]) if results else 0

    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke([prompt])

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    
    if return_context:
        output['context'] = context

    return output


class AdvancedRAGPipeline:
    """Advanced RAG pipeline with streaming, summarization, and history"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []
    
    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, 
              stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        """Execute advanced RAG query"""
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)

        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]

            prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {question}

Answer:"""
            
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            
            response = self.llm.invoke([prompt])
            answer = response.content

        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }
