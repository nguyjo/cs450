import sys
from pathlib import Path
import PyPDF2

sys.path.append(str(Path(__file__).parent.parent))

from kb import SimpleKnowledgeBase
from simple_rag import SimpleRAG

def parse_pdf_to_array(pdf_path):
    """Parse PDF pages into an array of strings."""
    pages = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page in reader.pages:
            text = page.extract_text()
            pages.append(text)
    
    return pages


def build_course_kb():
    """Build knowledge base with course information."""
    kb = SimpleKnowledgeBase()

    pdf_file = "syllabus.pdf"
    documents = parse_pdf_to_array(pdf_file)
    
    for doc in documents:
        kb.add_document(doc)
    
    return kb

def test_faq_system():
    """Test the FAQ system."""
    kb = build_course_kb()
    rag = SimpleRAG(kb)
    
    # TODO: Create test questions
    questions = [
        "What topics does CS450 cover?",
        "What are the CS450 student learning objectives?",
        "What is the CS450 Week of Worship?",
        "Tell me about the Week of Worship",
        "Tell me about the CS450 instructor",
        "Instructor:"
    ]
    
    print("Course FAQ System\n" + "="*50)
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 50)
        result = rag.answer_question(question)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. [{source['score']:.3f}] {source['text'][:60]}...")
        print("="*50)
        print()

if __name__ == "__main__":
    test_faq_system()