"""
Comprehensive preprocessing pipeline for document processing
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from ...core.models import DocumentContent, ExtractedContent
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator

from .metadata_extractor import MetadataExtractor
from .content_normalizer import ContentNormalizer, NormalizationOptions, NormalizationResult
from .temp_file_manager import TempFileManager


@dataclass
class PreprocessingOptions:
    """Options for preprocessing pipeline"""
    extract_comprehensive_metadata: bool = True
    normalize_content: bool = True
    preserve_original_content: bool = True
    enable_temp_file_management: bool = True
    normalization_options: Optional[NormalizationOptions] = None
    metadata_extraction_level: str = "comprehensive"  # basic, standard, comprehensive


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline"""
    success: bool
    processed_content: ExtractedContent
    original_content: Optional[ExtractedContent]
    comprehensive_metadata: Dict[str, Any]
    normalization_result: Optional[NormalizationResult]
    processing_time_ms: float
    warnings: List[str]
    errors: List[str]
    preprocessing_stats: Dict[str, Any]


class PreprocessingPipeline:
    """Comprehensive preprocessing pipeline"""
    
    def __init__(
        self, 
        options: Optional[PreprocessingOptions] = None,
        temp_file_manager: Optional[TempFileManager] = None
    ):
        self.options = options or PreprocessingOptions()
        self.temp_file_manager = temp_file_manager
        self.logger = get_pipeline_logger()
        
        # Initialize components
        self.metadata_extractor = MetadataExtractor()
        
        normalization_options = self.options.normalization_options or NormalizationOptions()
        self.content_normalizer = ContentNormalizer(normalization_options)
        
        self.logger.info("PreprocessingPipeline initialized", options=self.options)
    
    @timing_decorator
    def process(
        self, 
        document: DocumentContent, 
        extracted_content: ExtractedContent
    ) -> PreprocessingResult:
        """Run complete preprocessing pipeline"""
        
        start_time = datetime.now()
        warnings = []
        errors = []
        
        self.logger.info(
            f"Starting preprocessing pipeline for document {extracted_content.document_id}"
        )
        
        try:
            # Preserve original content if requested
            original_content = None
            if self.options.preserve_original_content:
                original_content = self._deep_copy_content(extracted_content)
            
            # Stage 1: Extract comprehensive metadata
            comprehensive_metadata = {}
            if self.options.extract_comprehensive_metadata:
                try:
                    comprehensive_metadata = self.metadata_extractor.extract_comprehensive_metadata(
                        document, extracted_content
                    )
                    self.logger.debug("Comprehensive metadata extraction completed")
                except Exception as e:
                    error_msg = f"Metadata extraction failed: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Stage 2: Content normalization
            normalization_result = None
            processed_content = extracted_content
            
            if self.options.normalize_content:
                try:
                    normalization_result = self.content_normalizer.normalize_content(extracted_content)
                    
                    # Update content with normalized version
                    processed_content = self._update_content_with_normalization(
                        extracted_content, normalization_result
                    )
                    
                    self.logger.debug(
                        "Content normalization completed",
                        changes_made=len(normalization_result.changes_made)
                    )
                    
                    if normalization_result.changes_made:
                        warnings.append(f"Content normalized with {len(normalization_result.changes_made)} changes")
                
                except Exception as e:
                    error_msg = f"Content normalization failed: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Stage 3: Update metadata with preprocessing information
            processing_metadata = self._create_processing_metadata(
                comprehensive_metadata, normalization_result, warnings, errors
            )
            
            # Merge all metadata
            final_metadata = self._merge_metadata(
                processed_content.metadata or {},
                comprehensive_metadata,
                processing_metadata
            )
            
            # Update processed content with final metadata
            processed_content.metadata = final_metadata
            
            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create preprocessing stats
            preprocessing_stats = self._create_preprocessing_stats(
                document, extracted_content, processed_content, 
                comprehensive_metadata, normalization_result
            )
            
            result = PreprocessingResult(
                success=len(errors) == 0,
                processed_content=processed_content,
                original_content=original_content,
                comprehensive_metadata=comprehensive_metadata,
                normalization_result=normalization_result,
                processing_time_ms=processing_time_ms,
                warnings=warnings,
                errors=errors,
                preprocessing_stats=preprocessing_stats
            )
            
            self.logger.info(
                f"Preprocessing pipeline completed for {extracted_content.document_id}",
                success=result.success,
                processing_time_ms=processing_time_ms,
                warnings_count=len(warnings),
                errors_count=len(errors)
            )
            
            return result
        
        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Preprocessing pipeline failed: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            return PreprocessingResult(
                success=False,
                processed_content=extracted_content,
                original_content=None,
                comprehensive_metadata={},
                normalization_result=None,
                processing_time_ms=processing_time_ms,
                warnings=warnings,
                errors=errors,
                preprocessing_stats={}
            )
    
    def _deep_copy_content(self, content: ExtractedContent) -> ExtractedContent:
        """Create a deep copy of extracted content"""
        return ExtractedContent(
            document_id=content.document_id,
            document_type=content.document_type,
            text_content=content.text_content,
            pages=content.pages.copy() if content.pages else None,
            sections=content.sections.copy() if content.sections else None,
            metadata=content.metadata.copy() if content.metadata else None,
            extraction_timestamp=content.extraction_timestamp
        )
    
    def _update_content_with_normalization(
        self, 
        original_content: ExtractedContent, 
        normalization_result: NormalizationResult
    ) -> ExtractedContent:
        """Update content with normalization results"""
        
        # Create new content with normalized text
        updated_content = ExtractedContent(
            document_id=original_content.document_id,
            document_type=original_content.document_type,
            text_content=normalization_result.normalized_content,
            pages=original_content.pages,
            sections=original_content.sections,
            metadata=original_content.metadata,
            extraction_timestamp=original_content.extraction_timestamp
        )
        
        # Normalize sections if they exist
        if original_content.sections:
            updated_content.sections = self.content_normalizer.normalize_sections(
                original_content.sections
            )
        
        return updated_content
    
    def _create_processing_metadata(
        self,
        comprehensive_metadata: Dict[str, Any],
        normalization_result: Optional[NormalizationResult],
        warnings: List[str],
        errors: List[str]
    ) -> Dict[str, Any]:
        """Create preprocessing-specific metadata"""
        
        processing_metadata = {
            'preprocessing': {
                'pipeline_version': '2.0',
                'processing_timestamp': datetime.now().isoformat(),
                'stages_completed': [],
                'warnings': warnings,
                'errors': errors,
                'success': len(errors) == 0
            }
        }
        
        # Add metadata extraction info
        if comprehensive_metadata:
            processing_metadata['preprocessing']['stages_completed'].append('metadata_extraction')
            processing_metadata['preprocessing']['metadata_extraction'] = {
                'extraction_level': self.options.metadata_extraction_level,
                'categories_extracted': list(comprehensive_metadata.keys()),
                'total_metadata_fields': self._count_metadata_fields(comprehensive_metadata)
            }
        
        # Add normalization info
        if normalization_result:
            processing_metadata['preprocessing']['stages_completed'].append('content_normalization')
            processing_metadata['preprocessing']['content_normalization'] = {
                'normalization_applied': True,
                'original_length': normalization_result.original_length,
                'normalized_length': normalization_result.normalized_length,
                'changes_made_count': len(normalization_result.changes_made),
                'changes_made': normalization_result.changes_made,
                'quality_improvements': normalization_result.quality_improvements,
                'normalization_summary': self.content_normalizer.get_normalization_summary(normalization_result)
            }
        
        return processing_metadata
    
    def _merge_metadata(
        self, 
        original_metadata: Dict[str, Any],
        comprehensive_metadata: Dict[str, Any],
        processing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge all metadata sources"""
        
        merged_metadata = original_metadata.copy()
        
        # Add comprehensive metadata
        for key, value in comprehensive_metadata.items():
            merged_metadata[f"comprehensive_{key}"] = value
        
        # Add processing metadata
        merged_metadata.update(processing_metadata)
        
        # Add metadata summary
        merged_metadata['metadata_summary'] = {
            'total_categories': len(comprehensive_metadata),
            'original_metadata_fields': len(original_metadata),
            'comprehensive_metadata_fields': self._count_metadata_fields(comprehensive_metadata),
            'processing_metadata_fields': self._count_metadata_fields(processing_metadata),
            'total_metadata_fields': self._count_metadata_fields(merged_metadata)
        }
        
        return merged_metadata
    
    def _count_metadata_fields(self, metadata: Dict[str, Any]) -> int:
        """Count total metadata fields recursively"""
        count = 0
        for value in metadata.values():
            if isinstance(value, dict):
                count += self._count_metadata_fields(value)
            else:
                count += 1
        return count
    
    def _create_preprocessing_stats(
        self,
        document: DocumentContent,
        original_content: ExtractedContent,
        processed_content: ExtractedContent,
        comprehensive_metadata: Dict[str, Any],
        normalization_result: Optional[NormalizationResult]
    ) -> Dict[str, Any]:
        """Create comprehensive preprocessing statistics"""
        
        stats = {
            'document_stats': {
                'original_size_bytes': document.size_bytes,
                'original_content_length': len(original_content.text_content or ""),
                'processed_content_length': len(processed_content.text_content or ""),
                'content_reduction_percentage': 0.0
            },
            'metadata_stats': {
                'comprehensive_metadata_extracted': bool(comprehensive_metadata),
                'metadata_categories': len(comprehensive_metadata),
                'total_metadata_fields': self._count_metadata_fields(comprehensive_metadata)
            },
            'normalization_stats': {},
            'quality_metrics': {},
            'processing_efficiency': {}
        }
        
        # Calculate content reduction
        original_length = len(original_content.text_content or "")
        processed_length = len(processed_content.text_content or "")
        
        if original_length > 0:
            reduction_percentage = ((original_length - processed_length) / original_length) * 100
            stats['document_stats']['content_reduction_percentage'] = round(reduction_percentage, 2)
        
        # Add normalization stats
        if normalization_result:
            stats['normalization_stats'] = {
                'normalization_applied': True,
                'changes_made': len(normalization_result.changes_made),
                'normalization_categories': list(normalization_result.normalization_stats.keys()),
                'quality_improvements': normalization_result.quality_improvements,
                'overall_improvement_score': sum(normalization_result.quality_improvements.values()) / max(len(normalization_result.quality_improvements), 1)
            }
        
        # Add quality metrics from comprehensive metadata
        if 'quality' in comprehensive_metadata:
            stats['quality_metrics'] = comprehensive_metadata['quality']
        
        # Add processing efficiency metrics
        stats['processing_efficiency'] = {
            'stages_completed': len([
                stage for stage in ['metadata_extraction', 'content_normalization'] 
                if (stage == 'metadata_extraction' and comprehensive_metadata) or 
                   (stage == 'content_normalization' and normalization_result)
            ]),
            'total_possible_stages': 2,
            'processing_success_rate': 1.0 if len(comprehensive_metadata) > 0 or normalization_result else 0.5
        }
        
        return stats
    
    def get_pipeline_summary(self, result: PreprocessingResult) -> Dict[str, Any]:
        """Get a summary of preprocessing pipeline results"""
        
        return {
            'success': result.success,
            'processing_time_ms': result.processing_time_ms,
            'warnings_count': len(result.warnings),
            'errors_count': len(result.errors),
            'stages_completed': result.preprocessing_stats.get('processing_efficiency', {}).get('stages_completed', 0),
            'content_changes': {
                'original_length': result.preprocessing_stats.get('document_stats', {}).get('original_content_length', 0),
                'processed_length': result.preprocessing_stats.get('document_stats', {}).get('processed_content_length', 0),
                'reduction_percentage': result.preprocessing_stats.get('document_stats', {}).get('content_reduction_percentage', 0.0)
            },
            'metadata_extracted': {
                'categories': result.preprocessing_stats.get('metadata_stats', {}).get('metadata_categories', 0),
                'total_fields': result.preprocessing_stats.get('metadata_stats', {}).get('total_metadata_fields', 0)
            },
            'normalization_applied': bool(result.normalization_result),
            'normalization_changes': len(result.normalization_result.changes_made) if result.normalization_result else 0,
            'quality_improvements': result.normalization_result.quality_improvements if result.normalization_result else {},
            'overall_quality_score': result.preprocessing_stats.get('quality_metrics', {}).get('overall_score', 0.0)
        }
    
    def validate_preprocessing_result(self, result: PreprocessingResult) -> Dict[str, Any]:
        """Validate preprocessing result and return validation report"""
        
        validation_report = {
            'is_valid': True,
            'validation_errors': [],
            'validation_warnings': [],
            'quality_checks': {}
        }
        
        # Check if processed content exists
        if not result.processed_content or not result.processed_content.text_content:
            validation_report['is_valid'] = False
            validation_report['validation_errors'].append("Processed content is empty")
        
        # Check if processing was successful
        if not result.success:
            validation_report['is_valid'] = False
            validation_report['validation_errors'].append("Processing reported as failed")
        
        # Check for critical errors
        if result.errors:
            validation_report['validation_warnings'].extend(result.errors)
        
        # Quality checks
        if result.processed_content and result.processed_content.text_content:
            content_length = len(result.processed_content.text_content)
            
            validation_report['quality_checks'] = {
                'content_length_adequate': content_length >= 10,
                'has_metadata': bool(result.comprehensive_metadata),
                'normalization_applied': bool(result.normalization_result),
                'processing_time_reasonable': result.processing_time_ms < 30000,  # 30 seconds
                'no_critical_errors': len(result.errors) == 0
            }
            
            # Check quality thresholds
            if content_length < 10:
                validation_report['validation_warnings'].append("Processed content is very short")
            
            if result.processing_time_ms > 30000:
                validation_report['validation_warnings'].append("Processing time exceeded 30 seconds")
        
        # Overall validation
        validation_report['is_valid'] = (
            validation_report['is_valid'] and 
            len(validation_report['validation_errors']) == 0 and
            all(validation_report['quality_checks'].values())
        )
        
        return validation_report