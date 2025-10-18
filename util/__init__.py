from .ollama_client import call_ollama, stream_ollama, chat_ollama

__all__ = ['call_ollama', 'stream_ollama', 'chat_ollama']

package_version = "1.0.0"

print(f"Initializing UTIL version {package_version}")