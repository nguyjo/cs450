import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import your SimpleKnowledgeBase
from kb import SimpleKnowledgeBase

# Create knowledge base
kb = SimpleKnowledgeBase()

# Add CS course information
documents = [
    "CS450 focuses on software engineering and AI applications.",
    "Data structures include arrays, linked lists, trees, and graphs.",
    "Algorithms course covers sorting, searching, and dynamic programming.",
    "Web development uses HTML, CSS, JavaScript, and frameworks like React.",
    "Databases store and organize data using SQL or NoSQL systems.",
    "Operating systems manage hardware and software resources.",
    "Computer networks enable communication between devices."
]

print("Adding Documents\n" + "="*50)
for doc in documents:
    kb.add_document(doc)

# Test queries
queries = [
    "What does CS450 cover?",
    "Tell me about data structures",
    "How do databases work?"
]

print("\n\nTesting Queries\n" + "="*50)
for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 50)
    results = kb.search(query, top_k=2)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['score']:.3f}] {result['text']}")