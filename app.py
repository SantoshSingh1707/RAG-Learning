import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import RAG components from core
from core import (
    process_all_pdf,
    split_document,
    EmbeddingManager,
    VectorStore,
    RAGRetrieval,
    rag_enhanced
)
from langchain_mistralai import ChatMistralAI

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        padding: 12px 20px;
    }
    .query-input {
        font-size: 16px;
    }
    .result-card {
        border-left: 4px solid #0066cc;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
        background-color: #f8f9fa;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    .score-high {
        background-color: #d4edda;
        color: #155724;
    }
    .score-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .score-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 RAG Question & Answer System")
st.markdown("Search your documents and get AI-powered answers using semantic search")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# Load RAG components only once using caching
@st.cache_resource
def load_rag_components():
    try:
        # Initialize components
        embedding_manager = EmbeddingManager(model_name="multi-qa-MiniLM-L6-cos-v1")
        vectorstore = VectorStore(
            collection_name="pdf_documents",
            persist_directory="data/vector_store"
        )
        vectorstore.initialize_store()
        
        retriever = RAGRetrieval(vectorstore, embedding_manager)
        llm = ChatMistralAI(model="mistral-small-2506", temperature=0.7)
        
        return retriever, llm, rag_enhanced
    except Exception as e:
        st.error(f"Error loading RAG components: {str(e)}")
        return None, None, None

# Sidebar parameters
top_k = st.sidebar.slider(
    "Number of documents to retrieve",
    min_value=1,
    max_value=10,
    value=5,
    help="How many relevant documents to search in"
)

score_threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Minimum similarity score to include results"
)

return_context = st.sidebar.checkbox(
    "Show source context",
    value=True,
    help="Display the original document snippets"
)

# Main search interface
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is attention all you need? What is machine learning?",
        key="search_query"
    )
with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# Load RAG components
retriever, llm, rag_enhanced_func = load_rag_components()

if retriever is None or llm is None:
    st.error("Failed to initialize RAG system. Please check your setup.")
else:
    # Handle search
    if search_button or (query and 'last_query' in st.session_state and st.session_state.last_query != query):
        if not query.strip():
            st.warning("Please enter a question to search.")
        else:
            st.session_state.last_query = query
            
            with st.spinner("🔍 Searching documents and generating answer..."):
                try:
                    # Get RAG response
                    result = rag_enhanced_func(
                        query=query,
                        retriever=retriever,
                        llm=llm,
                        top_k=top_k,
                        min_score=score_threshold,
                        return_context=return_context
                    )
                    
                    # Display answer
                    st.markdown("---")
                    st.subheader("💡 Answer")
                    st.markdown(result['answer'])
                    
                    # Display confidence
                    confidence = result.get('confidence', 0)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Display sources
                    sources = result.get('sources', [])
                    if sources:
                        st.subheader(f"📚 Retrieved Documents ({len(sources)})")
                        
                        for idx, source in enumerate(sources, 1):
                            with st.container():
                                # Create result card
                                st.markdown(f"### Result {idx}", help=f"Relevance: {source.get('similarity_score', 0):.2%}")
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    # File source
                                    file_source = source.get('source_file', 'Unknown')
                                    st.caption(f"📄 Source: {file_source}")
                                
                                with col2:
                                    # Similarity score badge
                                    score = source.get('similarity_score', 0)
                                    if score >= 0.7:
                                        badge_class = "score-high"
                                    elif score >= 0.5:
                                        badge_class = "score-medium"
                                    else:
                                        badge_class = "score-low"
                                    
                                    st.markdown(
                                        f'<span class="score-badge {badge_class}">{score:.1%}</span>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Document content
                                st.markdown(f"**Content Preview:**")
                                content = source.get('content', 'No content available')
                                # Truncate long content
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                st.text(content)
                                
                                st.divider()
                    else:
                        st.info("No relevant documents found for your query. Try adjusting the similarity threshold or asking a different question.")
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info("Make sure your vector store is populated with documents.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    RAG Q&A System | Powered by Mistral AI, ChromaDB & Streamlit
</div>
""", unsafe_allow_html=True)
