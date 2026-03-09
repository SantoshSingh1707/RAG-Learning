import streamlit as st
from pathlib import Path
import warnings
import tempfile
import os
import uuid
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

from src.data_loader import process_all_pdf, split_document, process_single_pdf, process_single_txt
from src.embedding import EmbeddingManager
from src.vector_store import VectorStore
from src.search import RAGRetrieval, rag_enhanced
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
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    .score-high { background-color: #d4edda; color: #155724; }
    .score-medium { background-color: #fff3cd; color: #856404; }
    .score-low { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 RAG Question & Answer System")
st.markdown("Search your documents and get AI-powered answers using semantic search")

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
        
        return retriever, llm, rag_enhanced, vectorstore, embedding_manager
    except Exception as e:
        st.error(f"Error loading RAG components: {str(e)}")
        return None, None, None, None, None

retriever, llm, rag_enhanced_func, vectorstore, embedding_manager = load_rag_components()

if retriever is None or llm is None:
    st.error("Failed to initialize RAG system. Please check your setup.")
    st.stop()

# --- Sidebar configuration ---
st.sidebar.title("⚙️ Configuration")

top_k = st.sidebar.slider("Number of documents to retrieve", 1, 10, 5)
score_threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.05)
return_context = st.sidebar.checkbox("Show source context", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT to Vector Store", type=["pdf", "txt"])
if uploaded_file:
    if st.sidebar.button("Process & Add Document", use_container_width=True):
        with st.sidebar.status("Processing document..."):
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write("Extracting text...")
            if uploaded_file.name.lower().endswith('.pdf'):
                docs = process_single_pdf(temp_path)
            elif uploaded_file.name.lower().endswith('.txt'):
                docs = process_single_txt(temp_path)
            else:
                docs = []
            
            if docs:
                st.write("Splitting into chunks...")
                chunks = split_document(docs)
                if chunks:
                    st.write("Generating embeddings...")
                    texts = [c.page_content for c in chunks]
                    embeddings = embedding_manager.generate_embeddings(texts)
                    st.write("Adding to database...")
                    vectorstore.add_documents(chunks, embeddings)
                    st.success(f"Successfully added {uploaded_file.name}!")
                else:
                    st.error("Document processed, but no extractable text chunks were found. This might be a scanned or image-based PDF.")
            else:
                st.error("Failed to extract text from document.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters & Management")
available_sources = vectorstore.get_available_sources()
selected_sources = st.sidebar.multiselect(
    "Filter by Source File",
    options=available_sources,
    default=[],
    help="Select specific documents to search in. Leave empty to search all."
)

if available_sources:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗑️ Remove Document")
    doc_to_remove = st.sidebar.selectbox("Select document to remove", [""] + available_sources)
    if doc_to_remove and st.sidebar.button("Delete Document", type="primary", use_container_width=True):
        with st.sidebar.status(f"Removing {doc_to_remove}..."):
            success = vectorstore.remove_source(doc_to_remove)
            if success:
                st.success(f"Successfully removed {doc_to_remove} from the database. Please restart or refresh the page to update the list if needed.")
                st.rerun()
            else:
                st.error("Failed to remove the document.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    RAG Q&A System | Powered by Mistral AI, ChromaDB & Streamlit
</div>
""", unsafe_allow_html=True)


# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 View {len(msg['sources'])} Sources"):
                for source in msg['sources']:
                    score = source.get('similarity_score', 0)
                    badge_class = "score-high" if score >= 0.7 else "score-medium" if score >= 0.5 else "score-low"
                    st.markdown(
                        f"**{source.get('source_file')}** (Page {source.get('page')}) <span class='score-badge {badge_class}'>{score:.1%}</span>", 
                        unsafe_allow_html=True
                    )
                    content = source.get('content', '')
                    st.caption(content[:300] + "..." if len(content) > 300 else content)
            
            dl_text = msg["content"] + "\n\nSources:\n" + "\n".join([f"- {s.get('source_file')} (Page {s.get('page')})" for s in msg["sources"]])
            st.download_button(
                "📥 Download Answer",
                data=dl_text,
                file_name=f"rag_answer_{msg['id']}.txt",
                mime="text/plain",
                key=f"dl_{msg['id']}"
            )

# Input
if query := st.chat_input("Ask a question about your documents..."):
    # Add passing user message
    st.session_state.messages.append({"role": "user", "content": query, "id": str(uuid.uuid4())})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = rag_enhanced_func(
                    query=query,
                    retriever=retriever,
                    llm=llm,
                    top_k=top_k,
                    min_score=score_threshold,
                    return_context=return_context,
                    source_filter=selected_sources if selected_sources else None
                )
                
                answer = result['answer']
                sources = result.get('sources', [])
                
                st.markdown(answer)
                if sources:
                    with st.expander(f"📚 View {len(sources)} Sources"):
                        for source in sources:
                            score = source.get('similarity_score', 0)
                            badge_class = "score-high" if score >= 0.7 else "score-medium" if score >= 0.5 else "score-low"
                            st.markdown(
                                f"**{source.get('source_file')}** (Page {source.get('page')}) <span class='score-badge {badge_class}'>{score:.1%}</span>", 
                                unsafe_allow_html=True
                            )
                            content = source.get('content', '')
                            st.caption(content[:300] + "..." if len(content) > 300 else content)
                    
                    msg_id = str(uuid.uuid4())
                    dl_text = answer + "\n\nSources:\n" + "\n".join([f"- {s.get('source_file')} (Page {s.get('page')})" for s in sources])
                    st.download_button(
                        "📥 Download Answer",
                        data=dl_text,
                        file_name=f"rag_answer_{msg_id}.txt",
                        mime="text/plain",
                        key=f"dl_{msg_id}"
                    )
                else:
                    msg_id = str(uuid.uuid4())
                    st.info("No relevant documents found for your query.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources, 
                    "id": msg_id
                })
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
