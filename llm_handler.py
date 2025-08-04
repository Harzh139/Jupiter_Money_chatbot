import os
from groq import Groq
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

class JupiterQABot:
    def __init__(self, groq_api_key=None):
        """Initialize the QA bot with Groq API"""
        if groq_api_key:
            self.groq_api_key = groq_api_key
        else:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found! Please set it in .env file or pass it directly")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.groq_api_key)
        
        # Default model
        self.model = "llama-3.3-70b-versatile"  # Fast and good quality
        
        print("✅ QA Bot initialized with Groq API")
    
    def create_context_prompt(self, question: str, search_results: List[Dict[str, Any]]) -> str:
        """Create a context-aware prompt for the LLM"""
        
        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            relevance_score = result.get('relevance_score', 0)
            if relevance_score > 0.3:  # Only include relevant results
                context_parts.append(f"Context {i}:\n{result['content']}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a helpful customer service assistant for Jupiter Money, a digital banking platform in India. 

Based on the following context information, answer the user's question accurately and helpfully.

CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. Be specific and helpful
3. If the context doesn't contain enough information, say so politely
4. Keep the tone friendly and professional
5. If asked about features, pricing, or processes, be as specific as possible
6. For account-related queries, guide them to contact support if needed

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, question: str, search_results: List[Dict[str, Any]], vector_store=None) -> Dict[str, Any]:
        """Generate an answer using Groq API and measure confidence by cosine similarity"""
        try:
            # Create the prompt
            prompt = self.create_context_prompt(question, search_results)
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=1000,
            )
            
            answer = chat_completion.choices[0].message.content

            # Calculate cosine similarity between question and top chunk
            if search_results and vector_store:
                question_emb = vector_store.embedding_model.encode([question])[0]
                top_chunk_emb = vector_store.embedding_model.encode([search_results[0]['content']])[0]
                confidence = float(cosine_similarity(question_emb, top_chunk_emb))
                confidence = max(0.0, min(confidence, 1.0))  # Clamp between 0 and 1
            else:
                confidence = 0.0

            return {
                'answer': answer,
                'confidence': confidence,
                'sources_used': len([r for r in search_results if r.get('relevance_score', 0) > 0.3]),
                'search_results': search_results
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}",
                'confidence': 0.0,
                'sources_used': 0,
                'search_results': search_results
            }
    
    def answer_question(self, question: str, vector_store, top_k: int = 5) -> Dict[str, Any]:
        """Complete pipeline: search + generate answer"""
        # Search for relevant context
        search_results = vector_store.search(question, top_k=top_k)
        
        # Generate answer
        response = self.generate_answer(question, search_results, vector_store=vector_store)
        
        # Add question to response
        response['question'] = question
        
        return response
    
    def chat_mode(self, vector_store):
        """Interactive chat mode for testing"""
        print("\n🤖 Jupiter Money QA Bot - Chat Mode")
        print("Ask me anything about Jupiter Money! (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🔍 Searching...")
            response = self.answer_question(question, vector_store)
            
            print(f"\n🤖 Answer (Confidence: {response['confidence']:.1%}):")
            print(response['answer'])
            
            if response['sources_used'] > 0:
                print(f"\n📚 Based on {response['sources_used']} relevant sources")

# Test function
def test_qa_bot():
    """Test the QA bot"""
    try:
        # Import vector store
        from vector_store import JupiterVectorStore
        
        # Initialize components
        print("🔄 Initializing components...")
        vector_store = JupiterVectorStore()
        
        # Load data and create embeddings if needed
        if not vector_store.load_data():
            print("❌ Cannot load data. Please run scraper.py first!")
            return False
        
        vector_store.create_embeddings()
        
        # Initialize QA bot (you'll need to set GROQ_API_KEY)
        qa_bot = JupiterQABot()
        
        # Test questions
        test_questions = [
            "What is Jupiter Money?",
            "How do I open an account?",
            "What are the main features?",
            "How can I contact support?"
        ]
        
        print("\n🧪 Testing QA Bot...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test {i} ---")
            response = qa_bot.answer_question(question, vector_store)
            
            print(f"Q: {question}")
            print(f"A: {response['answer'][:200]}...")
            print(f"Confidence: {response['confidence']:.1%}")
            print(f"Sources used: {response['sources_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing QA bot: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the QA bot
    success = test_qa_bot()
    
    if success:
        print("\n🎉 QA Bot test complete!")
    else:
        print("\n❌ QA Bot test failed!")
        print("🔧 Make sure you have:")
        print("   1. Set GROQ_API_KEY in .env file")
        print("   2. Run scraper.py first")
        print("   3. Run vector_store.py to create embeddings")