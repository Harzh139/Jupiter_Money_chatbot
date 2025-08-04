import os
import streamlit as st
import sys
from datetime import datetime
import json
import requests

# --- LangSmith/LangChain tracing env setup ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
# ---------------------------------------------

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from persistent_vector_store import PersistentVectorStore  # Updated import
    from llm_handler import JupiterQABot
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# LangSmith integration
import os
import uuid
try:
    from langsmith import trace
    LANGSMITH_API_KEY = st.secrets.get("LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY"))
    LANGSMITH_PROJECT = st.secrets.get("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "JupiterBot"))
except ImportError:
    trace = None
    LANGSMITH_API_KEY = None
    LANGSMITH_PROJECT = None

# Page configuration
st.set_page_config(
    page_title="Jupiter Money QA Bot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    color : #222 !important
}
.bot-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
    color : #222 !important
}
.confidence-high {
    color: #4caf50;
    font-weight: bold;
}
.confidence-medium {
    color: #ff9800;
    font-weight: bold;
}
.confidence-low {
    color: #f44336;
    font-weight: bold;
}
.storage-info {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_bot' not in st.session_state:
    st.session_state.qa_bot = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'question_submitted' not in st.session_state:
    st.session_state.question_submitted = False
if 'last_processed_question' not in st.session_state:
    st.session_state.last_processed_question = ""
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'embedding_status' not in st.session_state:
    st.session_state.embedding_status = "Not initialized"

@st.cache_resource
def initialize_persistent_vector_store(backend="chromadb"):
    """Initialize and cache the PERSISTENT vector store"""
    try:
        # Use persistent storage - embeddings will be saved to disk
        vs = PersistentVectorStore(
            backend=backend,
            persist_dir="./vector_db",  # This directory will be created and persist
            collection_name="jupiter_docs",
            force_rebuild=False  # Never force rebuild in production
        )
        return vs
    except Exception as e:
        st.error(f"❌ Error initializing persistent vector store: {str(e)}")
        return None

@st.cache_resource
def initialize_qa_bot():
    """Initialize and cache the QA bot using secrets"""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        return JupiterQABot(groq_api_key=groq_api_key)
    except Exception as e:
        st.error(f"❌ Error initializing QA bot: {str(e)}")
        return None

def setup_embeddings(vector_store):
    """Setup embeddings with proper status tracking"""
    if not vector_store:
        return False
    
    try:
        # Load data
        data_loaded = vector_store.load_data("data/prepared_data.json")
        if not data_loaded:
            st.session_state.embedding_status = "❌ No data file found"
            return False
        
        # Create embeddings (will be instant if they already exist)
        st.session_state.embedding_status = "🔄 Checking embeddings..."
        
        with st.spinner("🔄 Setting up embeddings (this may take time only on first run)..."):
            embeddings_created = vector_store.create_embeddings()
        
        if embeddings_created:
            stats = vector_store.get_stats()
            if stats['embeddings_exist']:
                st.session_state.embedding_status = f"✅ Ready ({stats['total_documents']} docs)"
                return True
            else:
                st.session_state.embedding_status = "❌ Embeddings creation failed"
                return False
        else:
            st.session_state.embedding_status = "❌ Failed to create embeddings"
            return False
            
    except Exception as e:
        st.session_state.embedding_status = f"❌ Error: {str(e)}"
        return False

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Jupiter Bot:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)

def set_sample_question(question):
    """Callback function to set a sample question"""
    st.session_state.current_question = question
    st.session_state.question_submitted = True

def run_with_langsmith(question, qa_bot, vector_store):
    """Wrap QA call with LangSmith tracing if available"""
    if trace and LANGSMITH_API_KEY:
        with trace(
            name="JupiterBot QA",
            project_name=LANGSMITH_PROJECT,
            api_key=LANGSMITH_API_KEY,
            tags=["jupiter", "qa-bot"],
            metadata={"question": question, "session_id": str(uuid.uuid4())}
        ) as run:
            response = qa_bot.answer_question(question, vector_store)
            run_id = getattr(run, "id", None)
            response["langsmith_run_id"] = str(run_id) if run_id else None
            return response
    else:
        response = qa_bot.answer_question(question, vector_store)
        response["langsmith_run_id"] = None
        return response

def send_langsmith_feedback(run_id, score, comment=""):
    """Send feedback to LangSmith for a given run_id"""
    api_key = st.secrets["LANGCHAIN_API_KEY"]
    url = f"https://api.smith.langchain.com/runs/{run_id}/feedback"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "score": score,
        "comment": comment
    }
    response = requests.post(url, headers=headers, json=data)
    return response.ok

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Jupiter Money QA Bot</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Backend selection
        backend_choice = st.selectbox(
            "Choose Storage Backend:",
            ["chromadb", "faiss", "pickle"],
            index=0,
            help="ChromaDB: Best for development, FAISS: Fastest search, Pickle: Simplest"
        )
        
        st.markdown("---")
        st.header("📊 System Status")
        
        # Data file check
        if os.path.exists("data/prepared_data.json"):
            st.success("✅ Data file found")
        else:
            st.error("❌ No data found")
            st.info("Please run the scraper first!")
        
        # Storage info
        st.markdown(f"""
        <div class="storage-info">
            <strong>🗄️ Persistent Storage</strong><br>
            Backend: {backend_choice.upper()}<br>
            Location: ./vector_db/<br>
            Status: {st.session_state.embedding_status}
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize components
        if not st.session_state.initialized:
            with st.spinner("🔄 Initializing system..."):
                # Initialize vector store
                st.session_state.vector_store = initialize_persistent_vector_store(backend_choice)
                
                if st.session_state.vector_store:
                    # Setup embeddings (persistent - fast after first run)
                    embeddings_ready = setup_embeddings(st.session_state.vector_store)
                    
                    if embeddings_ready:
                        # Initialize QA bot
                        st.session_state.qa_bot = initialize_qa_bot()
                        
                        if st.session_state.qa_bot:
                            st.session_state.initialized = True
                            st.success("✅ System ready!")
                        else:
                            st.error("❌ Failed to initialize QA bot")
                    else:
                        st.error("❌ Failed to setup embeddings")
                else:
                    st.error("❌ Failed to initialize vector store")
        else:
            st.success("✅ System ready!")
            if st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                st.info(f"📚 {stats['total_documents']} documents | {stats['backend'].upper()}")
                
                # Show persistent storage benefits
                if stats['embeddings_exist']:
                    st.success("⚡ Embeddings loaded from cache - instant startup!")
        
        # Management buttons
        st.markdown("---")
        st.header("🔧 Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data", help="Reload if data changed"):
                if st.session_state.vector_store:
                    st.session_state.vector_store.force_rebuild()
                    st.session_state.initialized = False
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache", help="Delete all cached embeddings"):
                if st.session_state.vector_store:
                    st.session_state.vector_store.clear_cache()
                    st.session_state.initialized = False
                    st.rerun()
        
        st.markdown("---")
        st.header("💡 Sample Questions")
        sample_questions = [
            "What is Jupiter Money?",
            "How do I open an account?",
            "What are the main features?",
            "What are the fees?",
            "How can I contact support?",
            "Is Jupiter Money safe?",
            "What cards do you offer?",
            "How do I make transactions?"
        ]
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                set_sample_question(question)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_processed_question = ""
            st.session_state.input_key += 1
            st.rerun()

    # Main content area
    # Performance info banner
    if st.session_state.initialized and st.session_state.vector_store:
        stats = st.session_state.vector_store.get_stats()
        if stats['embeddings_exist']:
            st.info("⚡ **Fast Mode Active**: Embeddings are cached and persistent. No re-processing on app restart!")
    
    # Tabs for Chat and Analytics
    tab1, tab2, tab3 = st.tabs(["Chat", "Analytics", "System Info"])

    with tab1:
        col1, _ = st.columns([3, 1])
        with col1:
            st.header("💬 Chat with Jupiter Bot")
            
            # Display chat history
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    display_chat_message(message['question'], is_user=True)
                    display_chat_message(message['answer'], is_user=False)
                    
                    # Show confidence
                    confidence = message.get('confidence', 0)
                    confidence_class = get_confidence_class(confidence)
                    st.markdown(f'<p class="{confidence_class}">Confidence: {confidence:.1%}</p>', 
                              unsafe_allow_html=True)
                    
                    # LangSmith tracing
                    if message.get("langsmith_run_id"):
                        run_id = message["langsmith_run_id"]
                        st.info(f"🔗 [View trace](https://smith.langchain.com/public/{run_id})")

                        with st.expander("📝 Give feedback"):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                thumbs_up = st.button("👍", key=f"up_{run_id}")
                                thumbs_down = st.button("👎", key=f"down_{run_id}")
                            with col2:
                                feedback_text = st.text_input("Optional comment", key=f"fb_{run_id}")

                            if thumbs_up or thumbs_down:
                                score = 1 if thumbs_up else 0
                                comment = feedback_text
                                if send_langsmith_feedback(run_id, score, comment):
                                    st.success("✅ Feedback sent!")
                                else:
                                    st.error("❌ Failed to send feedback")
                    
                    st.markdown("---")

            # Question input
            question_input = st.text_input(
                "Ask your question:",
                value=st.session_state.get('current_question', ''),
                placeholder="e.g., What is Jupiter Money?",
                key=f"question_input_{st.session_state.input_key}"
            )

            # Clear the current_question after using it
            if 'current_question' in st.session_state:
                del st.session_state.current_question

            col_send, col_clear = st.columns([1, 1])
            with col_send:
                send_button = st.button("📤 Send Question", type="primary")
            with col_clear:
                if st.button("🔄 New Question"):
                    st.session_state.input_key += 1
                    st.rerun()

            # Process question when button is clicked
            if send_button and question_input.strip():
                if question_input.strip() != st.session_state.last_processed_question:
                    if not st.session_state.initialized:
                        st.error("❌ System not ready. Please wait for initialization.")
                    else:
                        with st.spinner("🔍 Searching and generating answer..."):
                            try:
                                response = run_with_langsmith(
                                    question_input.strip(),
                                    st.session_state.qa_bot,
                                    st.session_state.vector_store
                                )
                                st.session_state.chat_history.append(response)
                                st.session_state.last_processed_question = question_input.strip()
                                st.session_state.input_key += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

            # Handle sample question submission
            if st.session_state.get('question_submitted', False):
                current_q = st.session_state.get('current_question', '')
                if current_q and current_q != st.session_state.last_processed_question:
                    if not st.session_state.initialized:
                        st.error("❌ System not ready. Please wait for initialization.")
                    else:
                        with st.spinner("🔍 Processing sample question..."):
                            try:
                                response = run_with_langsmith(
                                    current_q,
                                    st.session_state.qa_bot,
                                    st.session_state.vector_store
                                )
                                st.session_state.chat_history.append(response)
                                st.session_state.last_processed_question = current_q
                                st.session_state.input_key += 1
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                st.session_state.question_submitted = False
                if 'current_question' in st.session_state:
                    del st.session_state.current_question
                st.rerun()

    with tab2:
        st.header("📈 Analytics")
        if st.session_state.chat_history:
            # Calculate stats
            total_questions = len(st.session_state.chat_history)
            avg_confidence = sum(msg.get('confidence', 0) for msg in st.session_state.chat_history) / total_questions
            high_confidence_count = sum(1 for msg in st.session_state.chat_history if msg.get('confidence', 0) >= 0.7)

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", total_questions)
            with col2:
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            with col3:
                st.metric("High Confidence", f"{high_confidence_count}/{total_questions}")

            # Confidence distribution
            st.subheader("📊 Confidence Distribution")
            confidence_data = [msg.get('confidence', 0) for msg in st.session_state.chat_history]
            st.bar_chart(confidence_data)

            # Recent questions
            st.subheader("🕒 Recent Questions")
            for i, msg in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                confidence = msg.get('confidence', 0)
                confidence_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                st.write(f"{confidence_emoji} **Q{i}:** {msg['question'][:60]}...")
                st.write(f"   📈 Confidence: {confidence:.1%}")
                if msg.get("langsmith_run_id"):
                    st.write(f"   🔗 Trace: {msg['langsmith_run_id']}")
                st.write("")
        else:
            st.info("💬 Ask some questions to see analytics!")

    with tab3:
        st.header("🔧 System Information")
        
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Vector Store Stats")
                st.json({
                    "Backend": stats['backend'],
                    "Total Documents": stats['total_documents'],
                    "Embedding Model": stats['embedding_model'],
                    "Embedding Dimensions": stats['embedding_dimensions'],
                    "Embeddings Exist": stats['embeddings_exist'],
                    "Storage Location": stats['persist_dir']
                })
            
            with col2:
                st.subheader("💾 Storage Benefits")
                st.success("✅ **Persistent Storage Active**")
                st.info("""
                **Benefits:**
                - ⚡ Instant app startup after first run
                - 💾 Embeddings saved to disk
                - 🔄 Auto-detects data changes
                - 📈 Scales without re-processing
                - 🏗️ Production-ready architecture
                """)
                
                # Storage location info
                if os.path.exists("./vector_db"):
                    st.success("📁 Storage directory exists")
                    # Show directory size if possible
                    try:
                        import glob
                        files = glob.glob("./vector_db/*")
                        st.info(f"📁 {len(files)} cache files found")
                    except:
                        pass
                else:
                    st.warning("📁 Storage directory will be created on first run")
        
        # System requirements
        st.subheader("🔧 System Requirements")
        st.code("""
# Required packages for persistent storage:
pip install chromadb        # For ChromaDB backend
pip install faiss-cpu      # For FAISS backend (optional)
pip install sentence-transformers
pip install streamlit

# Storage requirements:
- Disk space: ~100MB for embeddings
- Memory: ~500MB during operation
- CPU: Any modern processor
        """)
        
        # Troubleshooting
        st.subheader("🩺 Troubleshooting")
        st.info("""
        **If embeddings keep recreating:**
        1. Check if `./vector_db/` directory persists
        2. Ensure write permissions to app directory
        3. Verify data file hasn't changed
        4. Try clearing cache and rebuilding
        
        **For deployment:**
        - Ensure persistent storage is mounted
        - Use volume mounts for Docker deployments
        - Consider read-only data after first setup
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏦 Jupiter Money QA Bot v2.0 | Now with Persistent Storage</p>
        <p><small>⚡ Embeddings cached for instant startup | Built with Streamlit, Groq API, and ChromaDB/FAISS</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()