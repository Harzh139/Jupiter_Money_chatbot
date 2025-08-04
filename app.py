import streamlit as st
import os
import sys
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from vector_store import JupiterVectorStore
    from llm_handler import JupiterQABot
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

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
def initialize_qa_bot(groq_api_key):
    """Initialize and cache the QA bot"""
    try:
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Jupiter Money QA Bot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Enter your Groq API key. Get one free at https://console.groq.com/keys"
        )
        
        st.markdown("---")
        
        # System status
        st.header("📊 System Status")
        
        # Check data availability
        if os.path.exists("data/prepared_data.json"):
            st.success("✅ Data loaded")
        else:
            st.error("❌ No data found")
            st.info("Please run `python scraper.py` first!")
        
        # Initialize components
        if groq_api_key:
            if not st.session_state.initialized:
                with st.spinner("🔄 Initializing components..."):
                    # Initialize vector store
                    st.session_state.vector_store = initialize_vector_store()
                    
                    # Initialize QA bot
                    if st.session_state.vector_store:
                        st.session_state.qa_bot = initialize_qa_bot(groq_api_key)
                        
                        if st.session_state.qa_bot:
                            st.session_state.initialized = True
                            st.success("✅ System ready!")
                        else:
                            st.error("❌ Failed to initialize QA bot")
                    else:
                        st.error("❌ Failed to initialize vector store")
            else:
                st.success("✅ System ready!")
                
                # Show stats
                if st.session_state.vector_store:
                    stats = st.session_state.vector_store.get_stats()
                    st.info(f"📚 {stats['total_documents']} documents loaded")
        else:
            st.warning("⚠️ Please enter your Groq API key")
        
        st.markdown("---")
        
        # Sample questions
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
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_processed_question = ""
            st.session_state.input_key += 1  # Also clear input
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat with Jupiter Bot")
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                display_chat_message(message['question'], is_user=True)
                
                # Bot response with confidence
                confidence = message.get('confidence', 0)
                confidence_class = get_confidence_class(confidence)
                
                bot_response = f"{message['answer']}"
                
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Jupiter Bot:</strong><br>
                    {bot_response}
                    <br><br>
                    <small>Confidence: <span class="{confidence_class}">{confidence:.1%}</span> | 
                    Sources: {message.get('sources_used', 0)}</small>
                </div>
                """, unsafe_allow_html=True)
        
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
            # Check if this is a different question from the last processed one
            if question_input.strip() != st.session_state.last_processed_question:
                if not st.session_state.initialized:
                    st.error("❌ Please configure the system first (enter API key in sidebar)")
                else:
                    with st.spinner("🔍 Searching and generating answer..."):
                        try:
                            # Get answer from QA bot
                            response = st.session_state.qa_bot.answer_question(
                                question_input.strip(), 
                                st.session_state.vector_store
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append(response)
                            st.session_state.last_processed_question = question_input.strip()
                            
                            # Increment key to create new input widget (clears the field)
                            st.session_state.input_key += 1
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error generating answer: {str(e)}")
        
        # Handle sample question submission (from sidebar buttons)
        if st.session_state.get('question_submitted', False):
            current_q = st.session_state.get('current_question', '')
            if current_q and current_q != st.session_state.last_processed_question:
                if not st.session_state.initialized:
                    st.error("❌ Please configure the system first (enter API key in sidebar)")
                else:
                    with st.spinner("🔍 Searching and generating answer..."):
                        try:
                            # Get answer from QA bot
                            response = st.session_state.qa_bot.answer_question(
                                current_q, 
                                st.session_state.vector_store
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append(response)
                            st.session_state.last_processed_question = current_q
                            
                            # Increment key to create new input widget (clears the field)
                            st.session_state.input_key += 1
                            
                        except Exception as e:
                            st.error(f"❌ Error generating answer: {str(e)}")
            
            # Reset the submission flag
            st.session_state.question_submitted = False
            if 'current_question' in st.session_state:
                del st.session_state.current_question
            st.rerun()
    
    with col2:
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
            
            # Recent questions
            st.subheader("Recent Questions")
            for i, msg in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                confidence = msg.get('confidence', 0)
                confidence_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                st.write(f"{confidence_emoji} {msg['question'][:50]}...")
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