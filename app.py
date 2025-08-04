import os
import streamlit as st
import sys
from datetime import datetime
import json
import requests

# --- LangSmith/LangChain tracing env setup (OPTIONAL) ---
try:
    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "JupiterBot")
except Exception as e:
    # LangSmith is optional - continue without it
    pass
# ---------------------------------------------

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from vector_store import JupiterVectorStore
    from llm_handler import JupiterQABot
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# LangSmith integration (OPTIONAL)
import os
import uuid
try:
    from langsmith import trace
    LANGSMITH_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY"))
    LANGSMITH_PROJECT = st.secrets.get("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "JupiterBot"))
    LANGSMITH_ENABLED = bool(LANGSMITH_API_KEY)
except (ImportError, KeyError, Exception):
    trace = None
    LANGSMITH_API_KEY = None
    LANGSMITH_PROJECT = None
    LANGSMITH_ENABLED = False

# Page configuration
st.set_page_config(
    page_title="Jupiter Money QA Bot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

@st.cache_resource
def initialize_vector_store():
    """Initialize and cache the vector store"""
    try:
        vs = JupiterVectorStore()
        if vs.load_data():
            vs.create_embeddings()
            return vs
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error initializing vector store: {str(e)}")
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
    try:
        if trace and LANGSMITH_ENABLED and LANGSMITH_API_KEY:
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
    except Exception as e:
        # If LangSmith fails, continue without tracing
        print(f"LangSmith tracing failed: {e}")
    
    # Fallback: run without LangSmith
    response = qa_bot.answer_question(question, vector_store)
    response["langsmith_run_id"] = None
    return response

def send_langsmith_feedback(run_id, score, comment=""):
    """Send feedback to LangSmith for a given run_id"""
    try:
        if not LANGSMITH_ENABLED:
            return False
        
        api_key = st.secrets.get("LANGCHAIN_API_KEY")
        if not api_key:
            return False
            
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
    except Exception as e:
        print(f"LangSmith feedback failed: {e}")
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Jupiter Money QA Bot</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Remove API Key input UI (do not prompt user for API key)
        st.markdown("---")
        st.header("📊 System Status")
        if os.path.exists("data/prepared_data.json"):
            st.success("✅ Data loaded")
        else:
            st.error("❌ No data found")
            st.info("Please run `python scraper.py` first!")
        # Initialize components
        if not st.session_state.initialized:
            with st.spinner("🔄 Initializing components..."):
                st.session_state.vector_store = initialize_vector_store()
                if st.session_state.vector_store:
                    st.session_state.qa_bot = initialize_qa_bot()
                    if st.session_state.qa_bot:
                        st.session_state.initialized = True
                        st.success("✅ System ready!")
                    else:
                        st.error("❌ Failed to initialize QA bot")
                else:
                    st.error("❌ Failed to initialize vector store")
        else:
            st.success("✅ System ready!")
            if st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                st.info(f"📚 {stats['total_documents']} documents loaded")
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
            st.session_state.input_key += 1  # Also clear input
            st.rerun()

    # Tabs for Chat and Analytics
    tab1, tab2 = st.tabs(["Chat", "Analytics"])

    with tab1:
        col1, _ = st.columns([3, 1])
        with col1:
            st.header("💬 Chat with Jupiter Bot")
            # Display chat history
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    display_chat_message(message['question'], is_user=True)
                    display_chat_message(message['answer'], is_user=False)
                    # Show confidence score and feedback section
                    confidence = message.get('confidence', 0)
                    confidence_class = get_confidence_class(confidence)
                    st.markdown(f'<p class="{confidence_class}">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
                    
                    # Always show feedback section (local feedback)
                    with st.expander("📝 Rate this answer"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            thumbs_up = st.button("👍 Helpful", key=f"up_{len(st.session_state.chat_history)}_{message.get('question', '')[:20]}")
                            thumbs_down = st.button("👎 Not helpful", key=f"down_{len(st.session_state.chat_history)}_{message.get('question', '')[:20]}")
                        with col2:
                            feedback_text = st.text_area("Optional comment (helps improve the bot):", 
                                                        key=f"fb_{len(st.session_state.chat_history)}_{message.get('question', '')[:20]}",
                                                        height=60)

                        if thumbs_up or thumbs_down:
                            score = 1 if thumbs_up else 0
                            comment = feedback_text.strip()
                            
                            # Store feedback locally
                            feedback_key = f"feedback_{len(st.session_state.chat_history)}"
                            if feedback_key not in st.session_state:
                                st.session_state[feedback_key] = {
                                    'score': score,
                                    'comment': comment,
                                    'question': message.get('question', ''),
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                # Try to send to LangSmith if available
                                langsmith_success = False
                                if message.get("langsmith_run_id") and LANGSMITH_ENABLED:
                                    langsmith_success = send_langsmith_feedback(message["langsmith_run_id"], score, comment)
                                
                                # Always show success message without mentioning LangSmith
                                st.success("✅ Thank you for your feedback! This helps improve the bot.")

            # Question input - using dynamic key to force refresh and clear
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
                    st.session_state.input_key += 1  # Force new input widget
                    st.rerun()

            # Process question when button is clicked
            if send_button and question_input.strip():
                if question_input.strip() != st.session_state.last_processed_question:
                    if not st.session_state.initialized:
                        st.error("❌ Please configure the system first (set secrets and run scraper)")
                    else:
                        with st.spinner("🔍 Searching and generating answer..."):
                            try:
                                # LangSmith tracing integration
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
                                st.error(f"❌ Error generating answer: {str(e)}")

            # Handle sample question submission (from sidebar buttons)
            if st.session_state.get('question_submitted', False):
                current_q = st.session_state.get('current_question', '')
                if current_q and current_q != st.session_state.last_processed_question:
                    if not st.session_state.initialized:
                        st.error("❌ Please configure the system first (set secrets and run scraper)")
                    else:
                        with st.spinner("🔍 Searching and generating answer..."):
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
                                st.error(f"❌ Error generating answer: {str(e)}")
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

            # Display stats
            st.metric("Total Questions", total_questions)
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
            st.metric("High Confidence Answers", f"{high_confidence_count}/{total_questions}")

            # Confidence distribution
            st.subheader("Confidence Distribution")
            confidence_data = [msg.get('confidence', 0) for msg in st.session_state.chat_history]
            st.bar_chart(confidence_data)

            # Feedback Analytics
            feedback_keys = [key for key in st.session_state.keys() if key.startswith('feedback_')]
            if feedback_keys:
                st.subheader("📊 User Feedback")
                positive_feedback = sum(1 for key in feedback_keys if st.session_state[key]['score'] == 1)
                total_feedback = len(feedback_keys)
                satisfaction_rate = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Feedback", total_feedback)
                with col2:
                    st.metric("Positive Feedback", positive_feedback)
                with col3:
                    st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
                
                # Recent feedback comments
                if any(st.session_state[key]['comment'] for key in feedback_keys):
                    st.subheader("💬 Recent Comments")
                    for key in reversed(feedback_keys[-3:]):
                        feedback = st.session_state[key]
                        if feedback['comment']:
                            emoji = "👍" if feedback['score'] == 1 else "👎"
                            st.write(f"{emoji} *{feedback['question'][:40]}...*")
                            st.write(f"   💭 \"{feedback['comment']}\"")
            
            # Recent questions
            st.subheader("Recent Questions")
            for i, msg in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                confidence = msg.get('confidence', 0)
                confidence_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                st.write(f"{confidence_emoji} {msg['question'][:50]}...")
            
            # Additional system info
            st.subheader("⚙️ System Info")
            st.info(f"📈 Total conversations: {len(st.session_state.chat_history)}")
            if st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                st.info(f"📚 Knowledge base: {stats['total_documents']} documents indexed")
        else:
            st.info("Ask a question to see analytics!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏦 Jupiter Money QA Bot | Built with Streamlit, Groq API, and ChromaDB</p>
        <p><small>This is a demo bot for assessment purposes</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()