import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import chat_ollama

def answer_as_role(question, role, expertise):
    """Answer questions from a specific role perspective."""
    system = f"You are a {role} with expertise in {expertise}."
    
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': question}
    ]
    
    return chat_ollama(messages, "gemma3", temperature=1.0)

question = "In one paragraph, how can I use software engineering to become very respected and/or wealthy?"

# Test different roles
roles = [
    ("university professor", "formal computer science education & academia"),
    ("senior software engineer", "cryptocurrency"),
    ("tech hobbyist", "GenAI"),
    ("FPGA Engineer", "UVM Verification")
]

for role, expertise in roles:
    print(f"\nRole: {role}")
    print(f"Expertise: {expertise}")
    print("-" * 60)
    answer = answer_as_role(question, role, expertise)
    print(answer)
    print("\n" + "="*60)