import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def compress_context(long_context):
    """Compress long context to key information."""
    prompt = f"""Extract only the essential information from this text.
Keep names, numbers, and key facts. Remove filler.

Text: {long_context}

Essential information:"""
    
    return call_ollama(prompt, temperature=0.2, num_predict=150)

long_text = """
Yesterday, I went to the grocery store, which was quite crowded 
actually, and I ran into my old friend Sarah Johnson. We chatted for 
a while about various things. She mentioned she got a new job at 
TechCorp starting next Monday with a salary of $85,000. She seemed 
really excited about it. The position is Senior Data Analyst. We 
agreed to meet for coffee next Friday at 3pm at Starbucks downtown.
"""

print("\n" + "="*60)
print("\nContext Compression:")
print("\nOriginal:", long_text)
print("\nCompressed:", compress_context(long_text))