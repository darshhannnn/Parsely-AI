"""
LLM response parsing and validation utilities
"""

import json
import re
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...core.logging_utils import get_pipeline_logger
from ...core.exceptions import LLMProcessingError
from .llm_integration import LLMResponse


class ResponseFormat(Enum):
    """Expected response formats"""
    JSON = "json"
    STRUCTURED_TEXT = "structured_text"
    PLAIN_TEXT = "plain_text"


@dataclass
class ParsedLLMResponse:
    """Represents a parsed and validated LLM response"""
    raw_response: str
    parsed_data: Dict[str, Any]
    format: ResponseFormat
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    confidence: float = 1.0


class LLMResponseParser:
    """Utility for parsing and validating LLM responses"""

    def __init__(self):
        self.logger = get_pipeline_logger()

    def parse_response(self, response: LLMResponse, expected_format: ResponseFormat = ResponseFormat.JSON) -> ParsedLLMResponse:
        """Parse LLM response into structured format"""
        raw_content = response.content
        errors = []
        parsed_data = {}
        is_valid = True

        if expected_format == ResponseFormat.JSON:
            try:
                parsed_data = self._extract_json(raw_content)
                if not parsed_data:
                    is_valid = False
                    errors.append("No valid JSON found in response")
            except Exception as e:
                is_valid = False
                errors.append(f"JSON parsing error: {str(e)}")

        elif expected_format == ResponseFormat.STRUCTURED_TEXT:
            parsed_data = self._parse_structured_text(raw_content)

        else:
            parsed_data = {"text": raw_content}

        return ParsedLLMResponse(
            raw_response=raw_content,
            parsed_data=parsed_data,
            format=expected_format,
            is_valid=is_valid,
            errors=errors
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text, handling potential markdown blocks"""
        # Try to find JSON in markdown blocks
        json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_blocks:
            return json.loads(json_blocks[0])

        # Try to find JSON between curly braces
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to parse entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text (e.g., key-value pairs)"""
        result = {}
        lines = text.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip().lower().replace(' ', '_')] = value.strip()
        return result

    def validate_schema(self, data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validate that all required fields are present in the data"""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, [f"Missing required fields: {', '.join(missing_fields)}"]
        return True, []