import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def analyze_text_chain(text):
    """Chain prompts to analyze text progressively."""
    
    # Step 1: Summarize
    summary_prompt = f"Summarize this in one sentence:\n\n{text}"
    summary = call_ollama(summary_prompt, temperature=0.3, num_predict=100)
    print("Step 1 - Summary:")
    print(summary)
    
    # Step 2: Extract key points from summary
    keypoints_prompt = f"List 3 key points from: {summary}"
    keypoints = call_ollama(keypoints_prompt, temperature=0.3)
    print("\nStep 2 - Key Points:")
    print(keypoints)
    
    # Step 3: Generate questions
    questions_prompt = f"Generate 2 questions someone might ask about:\n{keypoints}"
    questions = call_ollama(questions_prompt, temperature=0.5)
    print("\nStep 3 - Questions:")
    print(questions)
    
    return {
        'summary': summary,
        'keypoints': keypoints,
        'questions': questions
    }

# Test text
article = """
Artificial intelligence is transforming healthcare through advanced 
diagnostic tools. Machine learning algorithms can now detect diseases 
from medical images with accuracy matching expert radiologists. This 
technology reduces diagnosis time and helps doctors make better treatment 
decisions. However, challenges remain in data privacy and algorithm bias.
"""

print("Chained Analysis:\n" + "="*60 + "\n")
result = analyze_text_chain(article)