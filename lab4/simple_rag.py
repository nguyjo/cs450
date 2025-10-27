import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from kb import SimpleKnowledgeBase
from util import call_ollama

class SimpleRAG:
    """Simple RAG system combining retrieval and generation."""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def answer_question(self, question):
        """Answer question using RAG."""
        # Step 1: Retrieve relevant documents
        results = self.kb.search(question, top_k=2)
        
        # Step 2: Build context from retrieved docs
        context = "\n".join([r['text'] for r in results])
        
        # Step 3: Create prompt with context
        prompt = f"""Answer the question based on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
        
        # Step 4: Generate answer
        answer = call_ollama(prompt, temperature=0.2, num_predict=150)
        
        return {
            'answer': answer,
            'sources': results
        }

# Test RAG
if __name__ == "__main__":
    # Build knowledge base
    kb = SimpleKnowledgeBase()
    
    documents = [
        "Python was created by Guido van Rossum and released in 1991.",
        "Python emphasizes code readability with significant indentation.",
        "Python supports multiple programming paradigms including OOP.",
        "Python is used in web development, data science, and automation.",
        "JavaScript was created by Brendan Eich in 1995 for web browsers.",
        "JavaScript is the primary language for client-side web development."
    ]
    
    print("Building Knowledge Base\n" + "="*50)
    for doc in documents:
        kb.add_document(doc)
    
    # Create RAG system
    rag = SimpleRAG(kb)
    
    # Test questions
    questions = [
        "Who created Python?",
        "What is Python used for?",
        "When was JavaScript created?",
        "What is Python's execution speed?"  # Not in KB
    ]
    
    print("\n\nTesting RAG System\n" + "="*50)
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 50)
        result = rag.answer_question(question)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. [{source['score']:.3f}] {source['text'][:60]}...")
        print("="*50)