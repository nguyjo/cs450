import sys
from pathlib import Path
import ollama
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))

def get_embedding(text):
    """Get embedding vector for text."""
    client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')
    response = client.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return response['embedding']

# Compare similar and different texts
text1 = "He wore a black coat"
text2 = "She wore a blue vest"
text3 = "Python is a programming language"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# Calculate similarities
sim_1_2 = cosine_similarity([emb1], [emb2])[0][0]
sim_1_3 = cosine_similarity([emb1], [emb3])[0][0]

print("Semantic Similarity Test\n" + "="*50)
print(f"\nText 1: {text1}")
print(f"Text 2: {text2}")
print(f"Similarity: {sim_1_2:.4f}")

print(f"\nText 1: {text1}")
print(f"Text 3: {text3}")
print(f"Similarity: {sim_1_3:.4f}")