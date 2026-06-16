"""
Pipeline stages for the 6-stage document processing system
"""

from .stage1_input_documents import DocumentDownloader
from .stage2_llm_parser import Stage2LLMParser
from .stage3_embedding_search import UnifiedSearchEngine
from .stage4_clause_matching import ClauseMatcher
from .stage5_logic_evaluation import DecisionEngine
from .stage6_json_output import ResponseFormatter

__all__ = [
    'DocumentDownloader',
    'Stage2LLMParser',
    'UnifiedSearchEngine',
    'ClauseMatcher',
    'DecisionEngine',
    'ResponseFormatter'
]

__version__ = "2.0.0"