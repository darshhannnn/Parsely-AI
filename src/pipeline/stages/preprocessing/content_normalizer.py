"""
Content normalization and preprocessing utilities
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ...core.models import ExtractedContent
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator


@dataclass
class NormalizationOptions:
    """Options for content normalization"""
    normalize_whitespace: bool = True
    normalize_unicode: bool = True
    normalize_line_endings: bool = True
    remove_extra_spaces: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    normalize_case: bool = False
    remove_control_characters: bool = True
    preserve_formatting: bool = True
    language_specific_normalization: bool = True


@dataclass
class NormalizationResult:
    """Result of content normalization"""
    normalized_content: str
    original_length: int
    normalized_length: int
    changes_made: List[str]
    normalization_stats: Dict[str, int]
    quality_improvements: Dict[str, float]


class ContentNormalizer:
    """Content normalization and preprocessing"""
    
    def __init__(self, options: Optional[NormalizationOptions] = None):
        self.options = options or NormalizationOptions()
        self.logger = get_pipeline_logger()
    
    @timing_decorator
    def normalize_content(self, content: ExtractedContent) -> NormalizationResult:
        """Normalize extracted content"""
        
        self.logger.info(f"Starting content normalization for document {content.document_id}")
        
        original_text = content.text_content or ""
        original_length = len(original_text)
        
        if not original_text.strip():
            return NormalizationResult(
                normalized_content="",
                original_length=0,
                normalized_length=0,
                changes_made=[],
                normalization_stats={},
                quality_improvements={}
            )
        
        normalized_text = original_text
        changes_made = []
        stats = {}
        
        # Apply normalization steps
        if self.options.remove_control_characters:
            normalized_text, removed_count = self._remove_control_characters(normalized_text)
            if removed_count > 0:
                changes_made.append(f"Removed {removed_count} control characters")
                stats['control_characters_removed'] = removed_count
        
        if self.options.normalize_unicode:
            normalized_text, unicode_changes = self._normalize_unicode(normalized_text)
            if unicode_changes > 0:
                changes_made.append(f"Normalized {unicode_changes} Unicode characters")
                stats['unicode_normalizations'] = unicode_changes
        
        if self.options.normalize_line_endings:
            normalized_text, line_ending_changes = self._normalize_line_endings(normalized_text)
            if line_ending_changes > 0:
                changes_made.append(f"Normalized {line_ending_changes} line endings")
                stats['line_ending_normalizations'] = line_ending_changes
        
        if self.options.normalize_whitespace:
            normalized_text, whitespace_changes = self._normalize_whitespace(normalized_text)
            if whitespace_changes > 0:
                changes_made.append(f"Normalized {whitespace_changes} whitespace sequences")
                stats['whitespace_normalizations'] = whitespace_changes
        
        if self.options.remove_extra_spaces:
            normalized_text, space_changes = self._remove_extra_spaces(normalized_text)
            if space_changes > 0:
                changes_made.append(f"Removed {space_changes} extra spaces")
                stats['extra_spaces_removed'] = space_changes
        
        if self.options.normalize_quotes:
            normalized_text, quote_changes = self._normalize_quotes(normalized_text)
            if quote_changes > 0:
                changes_made.append(f"Normalized {quote_changes} quote characters")
                stats['quotes_normalized'] = quote_changes
        
        if self.options.normalize_dashes:
            normalized_text, dash_changes = self._normalize_dashes(normalized_text)
            if dash_changes > 0:
                changes_made.append(f"Normalized {dash_changes} dash characters")
                stats['dashes_normalized'] = dash_changes
        
        if self.options.normalize_case:
            normalized_text, case_changes = self._normalize_case(normalized_text)
            if case_changes > 0:
                changes_made.append(f"Normalized case for {case_changes} characters")
                stats['case_normalizations'] = case_changes
        
        if self.options.language_specific_normalization:
            normalized_text, lang_changes = self._apply_language_specific_normalization(
                normalized_text, content
            )
            if lang_changes > 0:
                changes_made.append(f"Applied {lang_changes} language-specific normalizations")
                stats['language_normalizations'] = lang_changes
        
        # Calculate quality improvements
        quality_improvements = self._calculate_quality_improvements(
            original_text, normalized_text
        )
        
        normalized_length = len(normalized_text)
        
        self.logger.info(
            f"Content normalization completed for {content.document_id}",
            original_length=original_length,
            normalized_length=normalized_length,
            changes_count=len(changes_made)
        )
        
        return NormalizationResult(
            normalized_content=normalized_text,
            original_length=original_length,
            normalized_length=normalized_length,
            changes_made=changes_made,
            normalization_stats=stats,
            quality_improvements=quality_improvements
        )
    
    def _remove_control_characters(self, text: str) -> Tuple[str, int]:
        """Remove control characters from text"""
        original_length = len(text)
        
        # Remove control characters except for common whitespace
        cleaned_text = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\n\r\t '
        )
        
        removed_count = original_length - len(cleaned_text)
        return cleaned_text, removed_count
    
    def _normalize_unicode(self, text: str) -> Tuple[str, int]:
        """Normalize Unicode characters"""
        original_text = text
        
        # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
        normalized_text = unicodedata.normalize('NFC', text)
        
        # Count changes (approximate)
        changes = sum(1 for a, b in zip(original_text, normalized_text) if a != b)
        
        return normalized_text, changes
    
    def _normalize_line_endings(self, text: str) -> Tuple[str, int]:
        """Normalize line endings to \n"""
        original_text = text
        
        # Replace Windows (\r\n) and Mac (\r) line endings with Unix (\n)
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Count changes
        changes = len(re.findall(r'\r\n|\r', original_text))
        
        return normalized_text, changes
    
    def _normalize_whitespace(self, text: str) -> Tuple[str, int]:
        """Normalize whitespace characters"""
        original_text = text
        
        # Replace various whitespace characters with standard space
        whitespace_chars = [
            '\u00A0',  # Non-breaking space
            '\u2000',  # En quad
            '\u2001',  # Em quad
            '\u2002',  # En space
            '\u2003',  # Em space
            '\u2004',  # Three-per-em space
            '\u2005',  # Four-per-em space
            '\u2006',  # Six-per-em space
            '\u2007',  # Figure space
            '\u2008',  # Punctuation space
            '\u2009',  # Thin space
            '\u200A',  # Hair space
            '\u202F',  # Narrow no-break space
            '\u205F',  # Medium mathematical space
            '\u3000',  # Ideographic space
        ]
        
        normalized_text = text
        changes = 0
        
        for ws_char in whitespace_chars:
            if ws_char in normalized_text:
                count = normalized_text.count(ws_char)
                normalized_text = normalized_text.replace(ws_char, ' ')
                changes += count
        
        return normalized_text, changes
    
    def _remove_extra_spaces(self, text: str) -> Tuple[str, int]:
        """Remove extra spaces while preserving intentional formatting"""
        original_text = text
        
        # Replace multiple spaces with single space, but preserve paragraph breaks
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Replace multiple spaces with single space within lines
            processed_line = re.sub(r' {2,}', ' ', line)
            processed_lines.append(processed_line)
        
        normalized_text = '\n'.join(processed_lines)
        
        # Count changes
        changes = len(original_text) - len(normalized_text)
        
        return normalized_text, max(changes, 0)
    
    def _normalize_quotes(self, text: str) -> Tuple[str, int]:
        """Normalize quote characters"""
        quote_mappings = {
            # Smart quotes to straight quotes
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '„': '"',  # Double low-9 quotation mark
            '‚': "'",  # Single low-9 quotation mark
            '«': '"',  # Left-pointing double angle quotation mark
            '»': '"',  # Right-pointing double angle quotation mark
            '‹': "'",  # Single left-pointing angle quotation mark
            '›': "'",  # Single right-pointing angle quotation mark
        }
        
        normalized_text = text
        changes = 0
        
        for old_quote, new_quote in quote_mappings.items():
            if old_quote in normalized_text:
                count = normalized_text.count(old_quote)
                normalized_text = normalized_text.replace(old_quote, new_quote)
                changes += count
        
        return normalized_text, changes
    
    def _normalize_dashes(self, text: str) -> Tuple[str, int]:
        """Normalize dash characters"""
        dash_mappings = {
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '―': '-',  # Horizontal bar
            '−': '-',  # Minus sign
        }
        
        normalized_text = text
        changes = 0
        
        for old_dash, new_dash in dash_mappings.items():
            if old_dash in normalized_text:
                count = normalized_text.count(old_dash)
                normalized_text = normalized_text.replace(old_dash, new_dash)
                changes += count
        
        return normalized_text, changes
    
    def _normalize_case(self, text: str) -> Tuple[str, int]:
        """Normalize case (if enabled)"""
        if not self.options.normalize_case:
            return text, 0
        
        # Simple case normalization - convert to title case for headings
        lines = text.split('\n')
        processed_lines = []
        changes = 0
        
        for line in lines:
            original_line = line
            
            # If line is all caps and looks like a heading, convert to title case
            if (line.isupper() and 
                len(line.split()) <= 10 and 
                not any(char.isdigit() for char in line)):
                processed_line = line.title()
                if processed_line != original_line:
                    changes += len([c for c in original_line if c.isupper()])
            else:
                processed_line = line
            
            processed_lines.append(processed_line)
        
        return '\n'.join(processed_lines), changes
    
    def _apply_language_specific_normalization(
        self, 
        text: str, 
        content: ExtractedContent
    ) -> Tuple[str, int]:
        """Apply language-specific normalization"""
        
        # Detect language from metadata or content
        language = self._detect_language(text, content)
        
        if language == 'english':
            return self._normalize_english_text(text)
        elif language == 'spanish':
            return self._normalize_spanish_text(text)
        elif language == 'french':
            return self._normalize_french_text(text)
        else:
            return text, 0
    
    def _detect_language(self, text: str, content: ExtractedContent) -> str:
        """Detect text language"""
        # Check if language is already detected in metadata
        if content.metadata:
            lang_info = content.metadata.get('content_analysis', {}).get('language', {})
            if lang_info.get('primary_language'):
                return lang_info['primary_language']
        
        # Simple language detection
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es']
        french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'en']
        
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
        spanish_count = sum(1 for word in spanish_words if f' {word} ' in f' {text_lower} ')
        french_count = sum(1 for word in french_words if f' {word} ' in f' {text_lower} ')
        
        if english_count >= spanish_count and english_count >= french_count:
            return 'english'
        elif spanish_count >= french_count:
            return 'spanish'
        elif french_count > 0:
            return 'french'
        else:
            return 'unknown'
    
    def _normalize_english_text(self, text: str) -> Tuple[str, int]:
        """Apply English-specific normalizations"""
        normalized_text = text
        changes = 0
        
        # Common English contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            if contraction in normalized_text:
                count = normalized_text.count(contraction)
                normalized_text = normalized_text.replace(contraction, expansion)
                changes += count
        
        return normalized_text, changes
    
    def _normalize_spanish_text(self, text: str) -> Tuple[str, int]:
        """Apply Spanish-specific normalizations"""
        # Spanish-specific normalizations would go here
        return text, 0
    
    def _normalize_french_text(self, text: str) -> Tuple[str, int]:
        """Apply French-specific normalizations"""
        # French-specific normalizations would go here
        return text, 0
    
    def _calculate_quality_improvements(self, original: str, normalized: str) -> Dict[str, float]:
        """Calculate quality improvements from normalization"""
        
        improvements = {}
        
        # Character consistency improvement
        original_unique_chars = len(set(original))
        normalized_unique_chars = len(set(normalized))
        if original_unique_chars > 0:
            improvements['character_consistency'] = max(
                0, (original_unique_chars - normalized_unique_chars) / original_unique_chars
            )
        
        # Whitespace consistency improvement
        original_whitespace_variety = len(set(char for char in original if char.isspace()))
        normalized_whitespace_variety = len(set(char for char in normalized if char.isspace()))
        if original_whitespace_variety > 0:
            improvements['whitespace_consistency'] = max(
                0, (original_whitespace_variety - normalized_whitespace_variety) / original_whitespace_variety
            )
        
        # Quote consistency improvement
        original_quote_variety = len(set(char for char in original if char in '""''„‚«»‹›'))
        normalized_quote_variety = len(set(char for char in normalized if char in '""''„‚«»‹›'))
        if original_quote_variety > 0:
            improvements['quote_consistency'] = max(
                0, (original_quote_variety - normalized_quote_variety) / original_quote_variety
            )
        
        # Overall readability improvement (simple heuristic)
        original_readability = self._calculate_simple_readability(original)
        normalized_readability = self._calculate_simple_readability(normalized)
        improvements['readability_improvement'] = max(0, normalized_readability - original_readability)
        
        return improvements
    
    def _calculate_simple_readability(self, text: str) -> float:
        """Calculate simple readability score"""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        # Simple readability heuristic (lower is better, normalize to 0-1)
        complexity = (avg_words_per_sentence / 20) + (avg_chars_per_word / 10)
        readability = max(0, 1 - min(complexity, 1))
        
        return readability
    
    def normalize_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Normalize content sections"""
        if not sections:
            return {}
        
        normalized_sections = {}
        
        for section_name, section_content in sections.items():
            if section_content:
                # Create temporary ExtractedContent for normalization
                temp_content = ExtractedContent(
                    document_id="temp",
                    document_type="temp",
                    text_content=section_content
                )
                
                result = self.normalize_content(temp_content)
                normalized_sections[section_name] = result.normalized_content
            else:
                normalized_sections[section_name] = section_content
        
        return normalized_sections
    
    def get_normalization_summary(self, result: NormalizationResult) -> Dict[str, Any]:
        """Get summary of normalization results"""
        
        reduction_percentage = 0.0
        if result.original_length > 0:
            reduction_percentage = (
                (result.original_length - result.normalized_length) / result.original_length * 100
            )
        
        return {
            'original_length': result.original_length,
            'normalized_length': result.normalized_length,
            'length_reduction_percentage': round(reduction_percentage, 2),
            'changes_made_count': len(result.changes_made),
            'changes_made': result.changes_made,
            'normalization_stats': result.normalization_stats,
            'quality_improvements': result.quality_improvements,
            'overall_improvement_score': sum(result.quality_improvements.values()) / max(len(result.quality_improvements), 1)
        }