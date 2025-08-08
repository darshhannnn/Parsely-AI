"""
Gemini API Configuration and Integration
"""

import os
from typing import Optional
import google.generativeai as genai
from loguru import logger
from typing import Any

class GeminiConfig:
    """Configuration class for Google Gemini API integration"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-pro")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        
        # Configure the Gemini client only if API key is present
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.warning("GOOGLE_API_KEY not set; Gemini features will be disabled")
        
    def get_llm_client(self) -> Any:
        """Get configured Gemini LLM client for LangChain"""
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY not configured")
        # Lazy import to avoid hard dependency at module import time
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            convert_system_message_to_human=True
        )
    
    def get_native_client(self) -> Optional[genai.GenerativeModel]:
        """Get native Gemini client for direct API calls"""
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY not configured")
        return genai.GenerativeModel(self.model_name)
    
    def test_connection(self) -> bool:
        """Test if Gemini API connection is working"""
        try:
            model = self.get_native_client()
            response = model.generate_content("Hello, this is a test.")
            logger.info("Gemini API connection successful")
            return True
        except Exception as e:
            logger.error(f"Gemini API connection failed: {str(e)}")
            return False

# Global instance
gemini_config = GeminiConfig()