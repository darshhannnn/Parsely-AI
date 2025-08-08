import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import pandas as pd

class InsuranceDocumentProcessor:
    def __init__(self, data_path: str = "data/policies"):
        self.data_path = data_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.processed_documents = []
    
    def get_file_paths(self) -> Dict[str, str]:
        """Get available PDF files from the data directory"""
        file_paths = {}
        
        if not os.path.exists(self.data_path):
            print(f"Warning: Directory {self.data_path} does not exist")
            return file_paths
        
        # Map actual files to friendly names
        file_mapping = {
            "ICIHLIP22012V012223.pdf": "ICICI_Golden_Shield",
            "HDFHLIP23024V072223.pdf": "HDFC_Easy_Health", 
            "BAJHLIP23020V012223.pdf": "Bajaj_Global_Health",
            "CHOTGDP23004V012223.pdf": "Cholamandalam_Travel",
            "EDLHLGA23009V012223.pdf": "Edelweiss_Maternity"
        }
        
        try:
            for filename in os.listdir(self.data_path):
                if filename.endswith('.pdf') and filename in file_mapping:
                    friendly_name = file_mapping[filename]
                    file_paths[friendly_name] = filename
        except Exception as e:
            print(f"Error reading directory {self.data_path}: {e}")
        
        return file_paths
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with robust error handling"""
        full_path = os.path.join(self.data_path, file_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} does not exist")
            return "Sample policy document for testing purposes."
        
        try:
            # Try PyPDF2 first
            import PyPDF2
            with open(full_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num} of {file_path}: {e}")
                        continue
                
                if text.strip():
                    return text
                else:
                    print(f"Warning: No text extracted from {file_path}, trying alternative method")
                    return self._extract_with_pdfplumber(full_path)
                    
        except Exception as e:
            print(f"Warning: PyPDF2 failed for {file_path}: {e}, trying alternative method")
            return self._extract_with_pdfplumber(full_path)
    
    def _extract_with_pdfplumber(self, full_path: str) -> str:
        """Alternative PDF extraction using pdfplumber"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(full_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Warning: pdfplumber failed on page {page_num}: {e}")
                        continue
            
            if text.strip():
                return text
            else:
                print(f"Warning: No text extracted with pdfplumber from {full_path}")
                return "Sample policy document for testing purposes."
                
        except ImportError:
            print("Warning: pdfplumber not available, using sample text")
            return "Sample policy document for testing purposes."
        except Exception as e:
            print(f"Warning: pdfplumber extraction failed: {e}")
            return "Sample policy document for testing purposes."

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        full_path = os.path.join(self.data_path, file_path)
        
        try:
            from docx import Document
            doc = Document(full_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Warning: Could not extract DOCX {file_path}: {e}")
            return "Sample policy document for testing purposes."

    def extract_text_from_eml(self, file_path: str) -> str:
        """Extract text from EML files"""
        full_path = os.path.join(self.data_path, file_path)
        
        try:
            import email
            with open(full_path, 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file)
                text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            text += part.get_payload(decode=True).decode('utf-8')
                else:
                    text = msg.get_payload(decode=True).decode('utf-8')
                return text
        except Exception as e:
            print(f"Warning: Could not extract EML {file_path}: {e}")
            return "Sample policy document for testing purposes."

    def extract_text(self, file_path: str) -> str:
        """Extract text based on file extension"""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            return self.extract_text_from_docx(file_path)
        elif ext == ".eml":
            return self.extract_text_from_eml(file_path)
        else:
            print(f"Warning: Unsupported file type: {ext}")
            return "Sample policy document for testing purposes."
