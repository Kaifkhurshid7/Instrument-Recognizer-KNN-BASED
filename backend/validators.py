"""
Validators module for input validation and custom exception handling.

This module provides:
- Custom exception classes for specific error scenarios
- File upload validators (format, size, duration)
- Audio format specifications
- Request/response data validation schemas

Author: Instrument Recognizer Team
Date: 2024
"""

from typing import Set, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError


# ============================================================================
# CUSTOM EXCEPTION CLASSES
# ============================================================================


class InstrumentRecognizerException(Exception):
    """
    Base exception class for all application-specific errors.
    
    All domain-specific exceptions inherit from this class for
    consistent error handling across the application.
    """
    pass


class AudioProcessingError(InstrumentRecognizerException):
    """
    Raised when audio file processing fails.
    
    Common causes:
    - Corrupted audio file
    - Unsupported codec
    - Memory exhaustion during processing
    """
    pass


class InvalidAudioFormatError(AudioProcessingError):
    """
    Raised when audio file format is not supported.
    
    Supported formats: mp3, wav, flac, m4a, ogg
    """
    pass


class AudioFileSizeError(AudioProcessingError):
    """
    Raised when audio file exceeds size limits.
    
    Default limit: 50MB
    """
    pass


class AudioDurationError(AudioProcessingError):
    """
    Raised when audio duration exceeds maximum allowed duration.
    
    Default limit: 600 seconds (10 minutes)
    """
    pass


class FeatureExtractionError(InstrumentRecognizerException):
    """
    Raised when spectral feature extraction fails.
    
    Common causes:
    - Invalid audio data
    - Corrupted waveform
    - Memory exhaustion
    """
    pass


class ModelNotReadyError(InstrumentRecognizerException):
    """
    Raised when classification is attempted before model initialization.
    
    Indicates model training/loading has not completed.
    """
    pass


class ClassificationError(InstrumentRecognizerException):
    """
    Raised when instrument classification fails.
    
    Indicates error during KNN/ML inference.
    """
    pass


# ============================================================================
# AUDIO CONFIGURATION CONSTANTS
# ============================================================================


class AudioConfig:
    """Configuration constants for audio file handling."""
    
    # Supported file formats
    SUPPORTED_FORMATS: Set[str] = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.m4b'}
    
    # File size limit (50 MB)
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024
    
    # Duration limit (10 minutes)
    MAX_DURATION_SECONDS: int = 600
    
    # Minimum duration (100ms)
    MIN_DURATION_SECONDS: float = 0.1
    
    # Sample rate for processing
    SAMPLE_RATE: int = 22050
    
    # Supported MIME types
    SUPPORTED_MIME_TYPES: Set[str] = {
        'audio/mpeg',
        'audio/wav',
        'audio/wav; codecs="1"',
        'audio/x-wav',
        'audio/flac',
        'audio/mp4',
        'audio/x-m4a',
        'audio/ogg',
        'application/octet-stream',
    }


# ============================================================================
# PYDANTIC VALIDATION MODELS
# ============================================================================


class AudioUploadRequest(BaseModel):
    """
    Validation schema for audio file uploads.
    
    Attributes:
        filename: Name of uploaded file
        file_size: Size of file in bytes
        mime_type: MIME type of file
    """
    
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., ge=1, le=AudioConfig.MAX_FILE_SIZE_BYTES)
    mime_type: Optional[str] = None
    
    @field_validator('filename')
    @classmethod
    def validate_filename_extension(cls, v: str) -> str:
        """
        Validate that file has supported audio extension.
        
        Args:
            v: Filename to validate
            
        Returns:
            Validated filename
            
        Raises:
            ValueError: If extension not supported
        """
        file_ext = Path(v).suffix.lower()
        if file_ext not in AudioConfig.SUPPORTED_FORMATS:
            supported = ', '.join(AudioConfig.SUPPORTED_FORMATS)
            raise ValueError(
                f"Unsupported format '{file_ext}'. "
                f"Supported formats: {supported}"
            )
        return v
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """
        Validate that file size is within acceptable limits.
        
        Args:
            v: File size in bytes
            
        Returns:
            Validated file size
            
        Raises:
            ValueError: If file exceeds size limit
        """
        max_mb = AudioConfig.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        if v > AudioConfig.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size ({v / (1024*1024):.1f}MB) exceeds "
                f"maximum allowed ({max_mb:.0f}MB)"
            )
        return v


class AnalysisConfig(BaseModel):
    """
    Configuration options for audio analysis.
    
    Attributes:
        include_waveform: Whether to return waveform visualization data
        confidence_threshold: Minimum confidence score (0-100)
        return_probabilities: Whether to return all class probabilities
    """
    
    include_waveform: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=100.0)
    return_probabilities: bool = Field(default=True)


class AnalysisResponse(BaseModel):
    """
    Standard response schema for analysis endpoint.
    
    Attributes:
        instrument: Predicted instrument class name
        confidence_score: Confidence percentage (0-100)
        waveform: Waveform visualization data (optional)
        feature_vector: 26-dimensional spectral feature vector
        compared_vector: Average feature vector for predicted class
        knn_probabilities: Probability distribution across all classes
    """
    
    instrument: str = Field(..., min_length=1)
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    waveform: Optional[dict] = None
    feature_vector: list = Field(default_factory=list)
    compared_vector: list = Field(default_factory=list)
    knn_probabilities: list = Field(default_factory=list)


class HealthCheckResponse(BaseModel):
    """
    Standard response schema for health check endpoint.
    
    Attributes:
        status: Service status ("ok" or "degraded")
        model_ready: Whether ML model is loaded and ready
        version: API version
    """
    
    status: str = Field(..., pattern='^(ok|degraded|error)$')
    model_ready: bool = Field(..., alias="model_ready")
    version: str = "2.0.0"
    
    class Config:
        protected_namespaces = ()


# ============================================================================
# VALIDATION HELPER FUNCTIONS
# ============================================================================


def validate_audio_file(
    filename: str,
    file_size: int,
    mime_type: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate audio file for upload.
    
    Args:
        filename: Name of file
        file_size: Size in bytes
        mime_type: MIME type (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_audio_file("song.mp3", 5000000)
        >>> if not is_valid:
        ...     print(f"Validation failed: {error}")
    """
    try:
        request = AudioUploadRequest(
            filename=filename,
            file_size=file_size,
            mime_type=mime_type
        )
        return True, None
    except ValidationError as e:
        error_messages = [error['msg'] for error in e.errors()]
        return False, '; '.join(error_messages)


def get_supported_formats() -> str:
    """
    Get human-readable list of supported formats.
    
    Returns:
        Comma-separated string of supported formats
        
    Example:
        >>> formats = get_supported_formats()
        >>> print(f"Supported: {formats}")
        Supported: .mp3, .wav, .flac, .m4a, .ogg
    """
    formats = sorted([fmt.upper().lstrip('.') for fmt in AudioConfig.SUPPORTED_FORMATS])
    return ', '.join(formats)
