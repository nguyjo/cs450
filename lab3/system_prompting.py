import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import chat_ollama

def test_system_prompt(user_query, system_message):
    """Test different system prompts."""
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_query}
    ]
    return chat_ollama(messages, temperature=0.3)

# Test 1: No system prompt (baseline)
query = "Explain quantum entanglement."
print("No System Prompt:")
print(test_system_prompt(query, ""))
print("\n" + "="*60 + "\n")

# Test 2: Concise system prompt
print("System: Be concise (max 2 sentences)")
result = test_system_prompt(
    query, 
    "You provide concise explanations. Maximum 2 sentences."
)
print(result)
print("\n" + "="*60 + "\n")

# Test 3: ELI5 system prompt
print("System: Explain like I'm 5")
result = test_system_prompt(
    query,
    "You explain complex topics to 5-year-olds using simple words and analogies."
)
print(result)