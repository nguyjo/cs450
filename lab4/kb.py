import sys
from pathlib import Path
import ollama
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))

class SimpleKnowledgeBase:
    """A simple document store with semantic search."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')
    
    def add_document(self, text):
        """Add a document to the knowledge base."""
        # Get embedding
        response = self.client.embeddings(
            model='nomic-embed-text',
            prompt=text
        )
        embedding = response['embedding']
        
        # Store document and embedding
        self.documents.append(text)
        self.embeddings.append(embedding)
        print(f"Added document: {text[:50]}...")
    
    def search(self, query, top_k=2):
        """Search for most relevant documents."""
        # Get query embedding
        response = self.client.embeddings(
            model='nomic-embed-text',
            prompt=query
        )
        query_embedding = response['embedding']
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding], 
            self.embeddings
        )[0]
        
        # Get top-k documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': similarities[idx]
            })
        
        return results

# Test the knowledge base
if __name__ == "__main__":
    kb = SimpleKnowledgeBase()
    
    # Add documents
    documents = [
        "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "JavaScript is primarily used for web development and runs in browsers.",
        "Machine learning is a subset of AI focused on learning from data.",
        "React is a JavaScript library for building user interfaces.",
        "Python is widely used in data science and machine learning."
    ]
    
    print("Building Knowledge Base\n" + "="*50)
    for doc in documents:
        kb.add_document(doc)
    
    # Test search
    query = "What is Python used for?"
    print(f"\n\nSearching for: {query}\n" + "="*50)
    results = kb.search(query)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   {result['text']}")