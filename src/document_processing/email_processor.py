import os
from typing import List
import email
from email import policy
from email.parser import BytesParser

class EmailDocumentProcessor:
    def __init__(self, data_path: str = "data/policies"):
        self.data_path = data_path
        self.processed_documents = []

    def get_file_paths(self) -> dict:
        # Placeholder: update with actual email files
        return {
            "Sample_Email_Policy": "sample_policy.eml"
        }

    def extract_text_from_eml(self, file_path: str) -> str:
        full_path = os.path.join(self.data_path, file_path)
        with open(full_path, 'rb') as fp:
            msg = BytesParser(policy=policy.default).parse(fp)
        # Extract plain text and, if present, HTML as fallback
        text = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == 'text/plain':
                    text.append(part.get_content())
        else:
            text.append(msg.get_content())
        # Fallback: get subject and from/to headers
        header_info = f"Subject: {msg.get('subject','')}\nFrom: {msg.get('from','')}\nTo: {msg.get('to','')}\n"
        return header_info + "\n".join(text)
