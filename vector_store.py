import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any

class JupiterVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize vector store with embedding model"""
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "jupiter_docs"
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print("✅ Loaded existing vector database")
        except:
            self.collection = self.client.create_collection(self.collection_name)
            print("✅ Created new vector database")
    
    def load_data(self, filepath="data/prepared_data.json"):
        """Load prepared data from JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"📚 Loaded {len(self.data)} text chunks")
            return True
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            print("🔄 Please run scraper.py first!")
            return False
    
    def create_embeddings(self):
        """Create embeddings for all text chunks"""
        if not hasattr(self, 'data'):
            print("❌ No data loaded. Please run load_data() first")
            return False
        
        # Check if embeddings already exist
        if self.collection.count() > 0:
            print("✅ Embeddings already exist in database")
            return True
        
        print("🔄 Creating embeddings... This might take a few minutes")
        
        # Extract texts for embedding
        texts = [item['content'] for item in self.data]
        ids = [item['id'] for item in self.data]
        
        # Create metadata
        metadatas = []
        for item in self.data:
            metadatas.append({
                'url': item['url'],
                'title': item['title'],
                'chunk_id': str(item['chunk_id'])
            })
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_tensor=False)
            all_embeddings.extend(batch_embeddings.tolist())
            print(f"✅ Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print("🎉 All embeddings created and stored!")
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Format results
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return search_results
    
    def get_stats(self):
        """Get database statistics"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
        }

# Test function
def test_vector_store():
    """Test the vector store functionality"""
    print("🧪 Testing Vector Store...")
    
    # Initialize
    vs = JupiterVectorStore()
    
    # Load data
    if not vs.load_data():
        return False
    
    # Create embeddings
    if not vs.create_embeddings():
        return False
    
    # Test search
    test_queries = [
        "What is Jupiter Money?",
        "How to open account?",
        "What are the features?",
        "Contact information"
    ]
    
    print("\n🔍 Testing search functionality:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vs.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['relevance_score']:.3f}")
            print(f"   Content: {result['content'][:100]}...")
            print(f"   Source: {result['metadata']['title']}")
    
    # Show stats
    stats = vs.get_stats()
    print(f"\n📊 Database Stats:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Embedding dimensions: {stats['embedding_model']}")
    
    return True

if __name__ == "__main__":
    # Create vector store and test
    success = test_vector_store()
    
    if success:
        print("\n🎉 Vector store setup complete!")
        print("✅ Ready for the next phase!")
    else:
        print("\n❌ Vector store setup failed!")
        print("🔧 Please check the error messages above")