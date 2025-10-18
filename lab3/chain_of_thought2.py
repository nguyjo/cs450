import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def solve_with_fewshot_cot(problem):
    """Use few-shot examples to guide reasoning."""
    prompt = f"""Solve word problems step by step.

Example:
Problem: A train travels 60 mph for 2 hours. How far does it go?
Solution:
Step 1: Speed = 60 mph
Step 2: Time = 2 hours
Step 3: Distance = Speed × Time = 60 × 2 = 120 miles
Answer: 120 miles

Example:
Problem: Jane has $50. She spends $15 on lunch and $20 on a book. How much remains?
Solution:
Step 1: Starting amount = $50
Step 2: Total spent = $15 + $20 = $35
Step 3: Remaining = $50 - $35 = $15
Answer: $15

Now solve just like my examples:
Problem: {problem}
Solution:"""
    
    return call_ollama(prompt, temperature=0.0)

problems = [
    "A rectangle is 8 cm long and 5 cm wide. What is its perimeter?",
    "Bob earns $25/hour and works 6 hours. He pays $30 in taxes. What's his take-home?"
]

print("\nFew-Shot CoT:")
for i, prob in enumerate(problems, 1):
    print(f"\nProblem {i}:")
    print(solve_with_fewshot_cot(prob))
    print("="*60)