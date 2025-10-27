import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from kb import SimpleKnowledgeBase
from simple_rag import SimpleRAG
from util import call_ollama

# Build knowledge base with specific information
kb = SimpleKnowledgeBase()

documents = [
    "Walla Walla University was founded in 1892 in College Place, Washington.",
    "WWU offers undergraduate and graduate programs in various fields.",
    "The university is affiliated with the Seventh-day Adventist Church.",
    "WWU's campus is located in the Walla Walla Valley, known for wine production."
]

print("Adding Documents to Knowledge Base\n" + "="*50)
for doc in documents:
    kb.add_document(doc)

# Create RAG system
rag = SimpleRAG(kb)

# Test question
question = "When was Walla Walla University founded?"

print("\n\nComparison: With and Without RAG\n" + "="*50)

# Without RAG (just LLM)
print("\nWITHOUT RAG (Direct LLM):")
print("-" * 50)
direct_prompt = f"Answer this question: {question}"
direct_answer = call_ollama(direct_prompt, temperature=0.2, num_predict=100)
print(f"Answer: {direct_answer}")

# With RAG
print("\n\nWITH RAG (Retrieved Context + LLM):")
print("-" * 50)
rag_result = rag.answer_question(question)
print(f"Answer: {rag_result['answer']}")
print(f"\nSources:")
for i, source in enumerate(rag_result['sources'], 1):
    print(f"  {i}. {source['text'][:70]}...")