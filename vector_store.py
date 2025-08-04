# --- SQLITE3 COMPATIBILITY FIX FOR CHROMADB ---
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass
# ---------------------------------------------

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class JupiterVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_dir="faiss_db"):
        """Initialize vector store with embedding model"""
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        self.data = []
        self.index = None
        self.id_map = []
        self.persist_dir = persist_dir
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Try to load existing index
        self._load_index()

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
        if not self.data:
            print("❌ No data loaded. Please run load_data() first")
            return False

        print("🔄 Creating embeddings for FAISS...")
        texts = [item['content'] for item in self.data]
        ids = [item['id'] for item in self.data]
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            embeddings = np.array(embeddings).astype('float32')

            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            self.id_map = ids
            
            # Save index to disk
            self._save_index()
            
            print(f"🎉 All embeddings created and stored in FAISS! ({len(texts)} documents, {dimension}D)")
            return True
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if self.index is None:
            print("❌ No FAISS index available. Please create embeddings first.")
            return []
            
        if not self.data:
            print("❌ No data loaded. Please load data first.")
            return []
            
        try:
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            D, I = self.index.search(query_embedding, min(top_k, len(self.data)))
            
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx == -1 or idx >= len(self.data):
                    continue
                    
                item = self.data[idx]
                results.append({
                    'content': item['content'],
                    'metadata': {
                        'url': item['url'],
                        'title': item['title'],
                        'chunk_id': item['chunk_id']
                    },
                    'distance': float(score),
                    'relevance_score': 1 / (1 + float(score))  # Convert L2 distance to a similarity-like score
                })
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {str(e)}")
            return []

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                # Save FAISS index
                index_path = os.path.join(self.persist_dir, "faiss_index.bin")
                faiss.write_index(self.index, index_path)
                
                # Save id_map and metadata
                metadata = {
                    'id_map': self.id_map,
                    'total_documents': len(self.data),
                    'embedding_dimension': self.index.d
                }
                metadata_path = os.path.join(self.persist_dir, "metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                    
                print(f"💾 FAISS index saved to {self.persist_dir}")
                return True
        except Exception as e:
            print(f"❌ Error saving index: {str(e)}")
            return False
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            index_path = os.path.join(self.persist_dir, "faiss_index.bin")
            metadata_path = os.path.join(self.persist_dir, "metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.id_map = metadata.get('id_map', [])
                    
                print(f"✅ FAISS index loaded from {self.persist_dir} ({len(self.id_map)} documents)")
                return True
        except Exception as e:
            print(f"⚠️ Could not load existing index: {str(e)}")
            return False
    
    def get_stats(self):
        """Get database statistics"""
        return {
            'total_documents': len(self.data),
            'backend': 'faiss',
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension(),
            'embeddings_exist': self.index is not None,
            'embedding_dimensions': self.index.d if self.index else None,
            'persist_dir': self.persist_dir,
            'index_size': self.index.ntotal if self.index else 0
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