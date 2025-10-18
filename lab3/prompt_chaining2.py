import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def refine_writing_chain(draft, style="professional"):
    """Progressively refine writing through multiple steps."""
    
    # Step 1: Fix grammar
    grammar_prompt = f"Fix any grammar errors:\n\n{draft}"
    fixed = call_ollama(grammar_prompt, temperature=0.2)
    print("Step 1 - Grammar Fixed:")
    print(fixed)
    
    # Step 2: Improve clarity
    clarity_prompt = f"Make this clearer and more concise:\n\n{fixed}"
    clear = call_ollama(clarity_prompt, temperature=0.3)
    print("\nStep 2 - Clarity Improved:")
    print(clear)
    
    # Step 3: Apply style
    style_prompt = f"Rewrite in {style} style:\n\n{clear}"
    final = call_ollama(style_prompt, temperature=0.4)
    print(f"\nStep 3 - {style.title()} Style:")
    print(final)
    
    return final

draft = """me and my team was working on the project 
but we didnt finish it because of there was problems"""

print("\n" + "="*60)
print("\nRefinement Chain:\n" + "="*60 + "\n")
refine_writing_chain(draft, "professional")