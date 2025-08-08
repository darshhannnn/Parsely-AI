"""
Gemini API Configuration and Integration
"""

import os
from typing import Optional
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

class GeminiConfig:
    """Configuration class for Google Gemini API integration"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-pro")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        
    def get_llm_client(self) -> ChatGoogleGenerativeAI:
        """Get configured Gemini LLM client for LangChain"""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            convert_system_message_to_human=True
        )
    
    def get_native_client(self):
        """Get native Gemini client for direct API calls"""
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