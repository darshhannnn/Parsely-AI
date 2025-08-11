"""
LLM response parsing and validation utilities
"""

import json
import re
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ...core.logging_utils import get_pipeline_logger
from ...core.exceptions import LLMProcessingError
from .llm_integration import LLMResponse


class ResponseFormat(Enum):
    """Expected response formats"""
    JSON = "json"
    STRUCTURED_TEXT = "structured_text"
    PLAIN_TEXT = "plain_text"