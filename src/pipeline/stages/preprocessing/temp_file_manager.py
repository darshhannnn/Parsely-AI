"""
Temporary file management and cleanup utilities
"""

import os
import tempfile
import shutil
import time
import threading
from typing import Dict, List, Optional, Set
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator


@dataclass
class TempFileInfo:
    """Information about a temporary file"""
    path: str
    created_at: datetime
    size_bytes: int
    purpose: str
    cleanup_after: Optional[datetime] = None
    auto_cleanup: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)


class TempFileManager:
    """Comprehensive temporary file management with automatic cleanup"""
    
    def __init__(
        self, 
        base_temp_dir: Optional[str] = None,
        max_age_hours: int = 24,
        max_total_size_mb: int = 1000,
        cleanup_interval_minutes: int = 30
    ):
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.max_age_hours = max_age_hours
        self.max_total_size_mb = max_total_size_mb
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        self.logger = get_pipeline_logger()
        self._temp_files: Dict[str, TempFileInfo] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Create managed temp directory
        self.managed_temp_dir = os.path.join(self.base_temp_dir, "llm_document_processor")
        os.makedirs(self.managed_temp_dir, exist_ok=True)
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        self.logger.info(
            f"TempFileManager initialized",
            managed_dir=self.managed_temp_dir,
            max_age_hours=max_age_hours,
            max_size_mb=max_total_size_mb
        )
    
    def _start_cleanup_thread(self):
        """Start the automatic cleanup thread"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="TempFileCleanup"
            )
            self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker for automatic cleanup"""
        while not self._shutdown_event.is_set():
            try:
                self.cleanup_expired_files()
                self.cleanup_oversized_cache()
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
            
            # Wait for next cleanup cycle
            self._shutdown_event.wait(timeout=self.cleanup_interval_minutes * 60)
    
    @timing_decorator
    def create_temp_file(
        self, 
        suffix: str = "", 
        prefix: str = "doc_", 
        purpose: str = "processing",
        auto_cleanup: bool = True,
        cleanup_after_hours: Optional[int] = None
    ) -> str:
        """Create a managed temporary file"""
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)  # milliseconds
        filename = f"{prefix}{timestamp}{suffix}"
        file_path = os.path.join(self.managed_temp_dir, filename)
        
        # Create the file
        with open(file_path, 'wb') as f:
            pass  # Create empty file
        
        # Calculate cleanup time
        cleanup_after = None
        if auto_cleanup:
            hours = cleanup_after_hours or self.max_age_hours
            cleanup_after = datetime.now() + timedelta(hours=hours)
        
        # Register the file
        file_info = TempFileInfo(
            path=file_path,
            created_at=datetime.now(),
            size_bytes=0,
            purpose=purpose,
            cleanup_after=cleanup_after,
            auto_cleanup=auto_cleanup
        )
        
        with self._lock:
            self._temp_files[file_path] = file_info
        
        self.logger.debug(
            f"Created temp file",
            path=file_path,
            purpose=purpose,
            auto_cleanup=auto_cleanup
        )
        
        return file_path
    
    @contextmanager
    def temp_file(
        self, 
        suffix: str = "", 
        prefix: str = "doc_", 
        purpose: str = "processing"
    ):
        """Context manager for temporary file"""
        file_path = self.create_temp_file(suffix, prefix, purpose, auto_cleanup=True)
        try:
            yield file_path
        finally:
            self.cleanup_file(file_path)
    
    def update_file_size(self, file_path: str) -> bool:
        """Update the size of a tracked file"""
        with self._lock:
            if file_path in self._temp_files:
                try:
                    size = os.path.getsize(file_path)
                    self._temp_files[file_path].size_bytes = size
                    return True
                except OSError:
                    return False
        return False
    
    def add_file_metadata(self, file_path: str, key: str, value: str) -> bool:
        """Add metadata to a tracked file"""
        with self._lock:
            if file_path in self._temp_files:
                self._temp_files[file_path].metadata[key] = value
                return True
        return False
    
    def cleanup_file(self, file_path: str) -> bool:
        """Clean up a specific temporary file"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                self.logger.debug(f"Cleaned up temp file: {file_path}")
            
            with self._lock:
                if file_path in self._temp_files:
                    del self._temp_files[file_path]
            
            return True
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
            return False
    
    def cleanup_expired_files(self) -> int:
        """Clean up expired temporary files"""
        now = datetime.now()
        expired_files = []
        
        with self._lock:
            for file_path, file_info in self._temp_files.items():
                if (file_info.auto_cleanup and 
                    file_info.cleanup_after and 
                    now > file_info.cleanup_after):
                    expired_files.append(file_path)
        
        cleaned_count = 0
        for file_path in expired_files:
            if self.cleanup_file(file_path):
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired temp files")
        
        return cleaned_count
    
    def cleanup_oversized_cache(self) -> int:
        """Clean up files if total cache size exceeds limit"""
        total_size_mb = self.get_total_size_mb()
        
        if total_size_mb <= self.max_total_size_mb:
            return 0
        
        # Get files sorted by age (oldest first)
        files_by_age = []
        with self._lock:
            for file_path, file_info in self._temp_files.items():
                if file_info.auto_cleanup:
                    files_by_age.append((file_info.created_at, file_path))
        
        files_by_age.sort()  # Sort by creation time
        
        cleaned_count = 0
        for _, file_path in files_by_age:
            if self.cleanup_file(file_path):
                cleaned_count += 1
                
                # Check if we're under the limit now
                if self.get_total_size_mb() <= self.max_total_size_mb:
                    break
        
        if cleaned_count > 0:
            self.logger.info(
                f"Cleaned up {cleaned_count} files to reduce cache size",
                original_size_mb=total_size_mb,
                new_size_mb=self.get_total_size_mb()
            )
        
        return cleaned_count
    
    def cleanup_all_files(self) -> int:
        """Clean up all managed temporary files"""
        file_paths = []
        with self._lock:
            file_paths = list(self._temp_files.keys())
        
        cleaned_count = 0
        for file_path in file_paths:
            if self.cleanup_file(file_path):
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up all {cleaned_count} temp files")
        return cleaned_count
    
    def get_total_size_mb(self) -> float:
        """Get total size of managed files in MB"""
        total_size = 0
        with self._lock:
            for file_info in self._temp_files.values():
                # Update size if file still exists
                if os.path.exists(file_info.path):
                    try:
                        current_size = os.path.getsize(file_info.path)
                        file_info.size_bytes = current_size
                        total_size += current_size
                    except OSError:
                        pass
        
        return total_size / (1024 * 1024)
    
    def get_file_count(self) -> int:
        """Get count of managed files"""
        with self._lock:
            return len(self._temp_files)
    
    def get_file_info(self, file_path: str) -> Optional[TempFileInfo]:
        """Get information about a managed file"""
        with self._lock:
            return self._temp_files.get(file_path)
    
    def list_files(self, purpose: Optional[str] = None) -> List[TempFileInfo]:
        """List managed files, optionally filtered by purpose"""
        files = []
        with self._lock:
            for file_info in self._temp_files.values():
                if purpose is None or file_info.purpose == purpose:
                    files.append(file_info)
        
        return files
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about managed files"""
        with self._lock:
            files = list(self._temp_files.values())
        
        if not files:
            return {
                'total_files': 0,
                'total_size_mb': 0.0,
                'oldest_file_age_hours': 0.0,
                'newest_file_age_hours': 0.0,
                'files_by_purpose': {},
                'auto_cleanup_enabled': 0,
                'files_pending_cleanup': 0
            }
        
        now = datetime.now()
        total_size = sum(f.size_bytes for f in files)
        ages = [(now - f.created_at).total_seconds() / 3600 for f in files]
        
        files_by_purpose = {}
        auto_cleanup_count = 0
        pending_cleanup = 0
        
        for file_info in files:
            # Count by purpose
            purpose = file_info.purpose
            files_by_purpose[purpose] = files_by_purpose.get(purpose, 0) + 1
            
            # Count auto cleanup
            if file_info.auto_cleanup:
                auto_cleanup_count += 1
                
                # Count pending cleanup
                if file_info.cleanup_after and now > file_info.cleanup_after:
                    pending_cleanup += 1
        
        return {
            'total_files': len(files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'oldest_file_age_hours': round(max(ages), 2),
            'newest_file_age_hours': round(min(ages), 2),
            'average_file_size_mb': round((total_size / len(files)) / (1024 * 1024), 2),
            'files_by_purpose': files_by_purpose,
            'auto_cleanup_enabled': auto_cleanup_count,
            'files_pending_cleanup': pending_cleanup,
            'managed_directory': self.managed_temp_dir
        }
    
    def force_cleanup_by_purpose(self, purpose: str) -> int:
        """Force cleanup of all files with specific purpose"""
        files_to_cleanup = []
        with self._lock:
            for file_path, file_info in self._temp_files.items():
                if file_info.purpose == purpose:
                    files_to_cleanup.append(file_path)
        
        cleaned_count = 0
        for file_path in files_to_cleanup:
            if self.cleanup_file(file_path):
                cleaned_count += 1
        
        self.logger.info(f"Force cleaned up {cleaned_count} files with purpose '{purpose}'")
        return cleaned_count
    
    def extend_file_lifetime(self, file_path: str, additional_hours: int) -> bool:
        """Extend the lifetime of a managed file"""
        with self._lock:
            if file_path in self._temp_files:
                file_info = self._temp_files[file_path]
                if file_info.cleanup_after:
                    file_info.cleanup_after += timedelta(hours=additional_hours)
                    self.logger.debug(
                        f"Extended lifetime of {file_path} by {additional_hours} hours"
                    )
                    return True
        return False
    
    def disable_auto_cleanup(self, file_path: str) -> bool:
        """Disable auto cleanup for a specific file"""
        with self._lock:
            if file_path in self._temp_files:
                self._temp_files[file_path].auto_cleanup = False
                self._temp_files[file_path].cleanup_after = None
                return True
        return False
    
    def shutdown(self):
        """Shutdown the temp file manager and cleanup"""
        self.logger.info("Shutting down TempFileManager")
        
        # Signal shutdown to cleanup thread
        self._shutdown_event.set()
        
        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Clean up all files
        self.cleanup_all_files()
        
        # Remove managed directory if empty
        try:
            if os.path.exists(self.managed_temp_dir):
                os.rmdir(self.managed_temp_dir)
        except OSError:
            pass  # Directory not empty or other error
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during destruction


# Global temp file manager instance
_global_temp_manager: Optional[TempFileManager] = None


def get_temp_file_manager() -> TempFileManager:
    """Get the global temp file manager instance"""
    global _global_temp_manager
    
    if _global_temp_manager is None:
        _global_temp_manager = TempFileManager()
    
    return _global_temp_manager


def create_managed_temp_file(
    suffix: str = "", 
    prefix: str = "doc_", 
    purpose: str = "processing"
) -> str:
    """Create a managed temporary file using global manager"""
    return get_temp_file_manager().create_temp_file(suffix, prefix, purpose)


def cleanup_managed_temp_file(file_path: str) -> bool:
    """Clean up a managed temporary file using global manager"""
    return get_temp_file_manager().cleanup_file(file_path)


@contextmanager
def managed_temp_file(suffix: str = "", prefix: str = "doc_", purpose: str = "processing"):
    """Context manager for managed temporary file"""
    with get_temp_file_manager().temp_file(suffix, prefix, purpose) as file_path:
        yield file_path