import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def solve_with_cot(problem):
    """Solve using Chain of Thought."""
    prompt = f"""{problem}

Let's solve this step by step:"""
    
    return call_ollama(prompt, temperature=0.0)


# Logic problem
problem = """A farmer is on one side of a river with a wolf, a goat, and a cabbage. When he is crossing the river in a boat, he can only take one item with him at a time. The wolf will eat the goat if left alone together, and the goat will eat the cabbage if left alone together. How can the farmer transport the goat across the river without it being eaten?"""

print("\n" + "="*60)
print("\nProblem (Logic):")
print(solve_with_cot(problem))