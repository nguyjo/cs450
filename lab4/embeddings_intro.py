import sys
from pathlib import Path
import ollama

sys.path.append(str(Path(__file__).parent.parent))

def get_embedding(text):
    """Get embedding vector for text."""
    client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')
    response = client.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return response['embedding']

# Test embeddings
texts = [
    "The cat sat on the mat",
    "A feline rested on a rug",
    "The dog played in the park"
]

print("Generating Embeddings\n" + "="*50)
for text in texts:
    embedding = get_embedding(text)
    print(f"\nText: {text}")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")