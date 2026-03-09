import time
from typing import List, Dict, Any

from src.vector_store import VectorStore
from src.embedding import EmbeddingManager

class RAGRetrieval:
    """Retrieves documents from vector store based on query similarity"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.35, source_filter: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar documents"""
        print(f"Retrieving documents for query : '{query}' ")
        print(f"Top_k : {top_k} , Score_threshold : {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query], is_query=True)[0]
        
        try:
            query_kwargs = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k
            }
            if source_filter:
                if len(source_filter) == 1:
                    query_kwargs["where"] = {"source_file": source_filter[0]}
                else:
                    query_kwargs["where"] = {"source_file": {"$in": source_filter}}
                    
            results = self.vector_store.collection.query(**query_kwargs)

            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # ChromaDB returns squared L2 distance. Typical matches are ~1.0 to ~1.4.
                    # We map this to a more intuitive 0-1 similarity score where 1.3 distance = ~85% confidence.
                    similarity_score = max(0.0, min(1.0, 1.5 - (distance / 2.0)))
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


def rag_enhanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False, source_filter=None):
    """Enhanced RAG pipeline with sources and confidence"""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score, source_filter=source_filter)
    
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
