"""
Simple document processor that doesn't crash on missing files
"""

import os
from typing import List, Dict

class SimpleDocumentProcessor:
    """Simple document processor for testing"""
    
    def __init__(self, data_path: str = "data/policies"):
        self.data_path = data_path
    
    def get_file_paths(self) -> Dict[str, str]:
        """Get available file paths"""
        files = {}
        
        if not os.path.exists(self.data_path):
            return files
        
        try:
            for filename in os.listdir(self.data_path):
                if filename.endswith(('.pdf', '.docx', '.txt')):
                    files[filename] = os.path.join(self.data_path, filename)
        except Exception as e:
            print(f"Warning: Could not read directory {self.data_path}: {e}")
        
        return files
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with error handling"""
        
        if not os.path.exists(file_path):
            return "Sample policy document text for testing purposes."
        
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Warning: Could not process PDF {file_path}: {e}")
            return "Sample policy document text for testing purposes."