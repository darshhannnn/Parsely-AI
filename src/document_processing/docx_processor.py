import os
from typing import List
from docx import Document

class DocxDocumentProcessor:
    def __init__(self, data_path: str = "data/policies"):
        self.data_path = data_path
        self.processed_documents = []

    def get_file_paths(self) -> dict:
        # Placeholder: update with actual docx policy files
        return {
            "Sample_Word_Policy": "sample_policy.docx"
        }

    def extract_text_from_docx(self, file_path: str) -> str:
        full_path = os.path.join(self.data_path, file_path)
        doc = Document(full_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
