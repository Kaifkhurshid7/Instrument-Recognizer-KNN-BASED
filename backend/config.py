"""
Centralized application configuration with type hints and validation.

Configuration is managed through:
1. Environment variables (highest priority)
2. Configuration class defaults (lowest priority)

This ensures:
- Type safety with mypy
- IDE autocompletion
- Single source of truth
- Easy deployment configuration

Environment variables:
    PORT: Server port (default: 5000)
    ENV: Environment name - 'development', 'staging', 'production'
    LOG_LEVEL: Logging level - 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    MAX_UPLOAD_SIZE_MB: Maximum file upload size in MB
    MAX_AUDIO_DURATION: Maximum audio duration in seconds

Author: Instrument Recognizer Team
Date: 2024
"""

import os
from pathlib import Path
from typing import Literal
from enum import Enum


# ============================================================================
# ENVIRONMENT AND PATH CONFIGURATION
# ============================================================================


class Environment(str, Enum):
    """
    Supported application environments.
    
    Determines behavior of logging, error handling, and caching.
    """
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# Base directory of the application
BASE_DIR: Path = Path(__file__).parent.resolve()

# Parent directory (project root)
PROJECT_ROOT: Path = BASE_DIR.parent.resolve()

# Environment detection
ENVIRONMENT: Environment = Environment(
    os.getenv("ENV", "development").lower()
)

IS_PRODUCTION: bool = ENVIRONMENT == Environment.PRODUCTION
IS_DEVELOPMENT: bool = ENVIRONMENT == Environment.DEVELOPMENT
IS_STAGING: bool = ENVIRONMENT == Environment.STAGING


# ============================================================================
# FILE PATH CONFIGURATION
# ============================================================================


class PathConfig:
    """File and directory paths configuration."""
    
    # Model and database files
    DATABASE_FILE: Path = BASE_DIR / "reference_database.pkl"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Upload and temporary files
    UPLOAD_FOLDER: Path = BASE_DIR / "uploads"
    TEMP_FOLDER: Path = BASE_DIR / "temp"
    
    # Training data
    DATASET_PATH: Path = BASE_DIR / "IRMAS-TrainingData"
    
    # Logs directory
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    @classmethod
    def ensure_directories_exist(cls) -> None:
        """
        Create all required directories if they don't exist.
        
        Raises:
            PermissionError: If directories cannot be created
        """
        for directory in [cls.UPLOAD_FOLDER, cls.TEMP_FOLDER, 
                          cls.MODELS_DIR, cls.LOGS_DIR]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                raise PermissionError(
                    f"Cannot create directory {directory}: {e}"
                )


# ============================================================================
# AUDIO PROCESSING CONFIGURATION
# ============================================================================


class AudioProcessingConfig:
    """Configuration for audio feature extraction and processing."""
    
    # Sample rate for librosa processing (Hz)
    # Standard for music information retrieval
    SAMPLE_RATE: int = 22050
    
    # Maximum audio duration for processing (seconds)
    # Prevents memory exhaustion on large files
    MAX_DURATION_SECONDS: int = int(
        os.getenv("MAX_AUDIO_DURATION", "600")
    )
    
    # Minimum audio duration required (seconds)
    # Ensures enough data for reliable feature extraction
    MIN_DURATION_SECONDS: float = 0.5
    
    # Number of samples to display in waveform visualization
    WAVEFORM_DISPLAY_SAMPLES: int = 1000
    
    # FFT window size for spectrogram
    N_FFT: int = 2048
    
    # Hop length for spectrogram (in samples)
    HOP_LENGTH: int = 512


# ============================================================================
# MACHINE LEARNING MODEL CONFIGURATION
# ============================================================================


class ModelConfig:
    """Configuration for ML model parameters."""
    
    # Feature vector dimensionality
    # Consists of: 13 MFCCs + 1 delta + 2 chroma + 2 centroid + 
    #              2 rolloff + 2 ZCR + 2 bandwidth + 1 flatness + 1 RMS = 26
    FEATURE_VECTOR_LENGTH: int = 26
    
    # Number of MFCC coefficients
    MFCC_COEFFICIENTS: int = 13
    
    # KNN classifier parameters
    KNN_NEIGHBORS: int = 3
    KNN_METRIC: Literal["cosine", "euclidean", "manhattan"] = "cosine"
    KNN_WEIGHTS: Literal["uniform", "distance"] = "distance"
    
    # List of instrument classes (must match training data)
    INSTRUMENT_CLASSES: tuple = (
        "acoustic guitar",
        "cello",
        "clarinet",
        "electric guitar",
        "flute",
        "human voice",
        "organ",
        "piano",
        "saxophone",
        "trumpet",
        "violin",
    )
    
    NUM_CLASSES: int = len(INSTRUMENT_CLASSES)


# ============================================================================
# SERVER AND API CONFIGURATION
# ============================================================================


class ServerConfig:
    """Configuration for FastAPI/Uvicorn server."""
    
    # Server port
    PORT: int = int(os.getenv("PORT", "5000"))
    
    # Server host
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # Number of worker processes (for production)
    # Auto-calculated based on CPU cores if 0
    WORKERS: int = int(os.getenv("WORKERS", "0"))
    
    # Request timeout in seconds
    REQUEST_TIMEOUT: int = 60
    
    # Uvicorn reload (development only)
    RELOAD: bool = IS_DEVELOPMENT
    
    # Debug mode
    DEBUG: bool = IS_DEVELOPMENT
    
    # API version
    API_VERSION: str = "2.0.0"
    
    # API title
    API_TITLE: str = "Instrument Recognizer API"
    
    # API description
    API_DESCRIPTION: str = (
        "AI-powered musical instrument classification from audio files. "
        "Supports 11 instrument classes with explainable results."
    )


# ============================================================================
# CORS AND SECURITY CONFIGURATION
# ============================================================================


class CorsConfig:
    """Cross-Origin Resource Sharing (CORS) configuration."""
    
    # Allowed origins for CORS
    # In production, should be restricted to specific domains
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Add production origins if available
    if IS_PRODUCTION:
        ALLOWED_ORIGINS.extend([
            "https://yourdomain.com",
            "https://www.yourdomain.com",
        ])
    
    # Allow credentials (cookies, authorization headers)
    ALLOW_CREDENTIALS: bool = True
    
    # Allowed HTTP methods
    ALLOW_METHODS: list = ["GET", "POST", "OPTIONS"]
    
    # Allowed headers
    ALLOW_HEADERS: list = ["*"]


class SecurityConfig:
    """Security-related configuration."""
    
    # Maximum file upload size (bytes)
    MAX_UPLOAD_SIZE_BYTES: int = int(
        os.getenv("MAX_UPLOAD_SIZE_MB", "50")
    ) * 1024 * 1024
    
    # Rate limit: requests per hour per IP
    RATE_LIMIT_PER_HOUR: int = 100
    
    # Rate limit: requests per minute per IP
    RATE_LIMIT_PER_MINUTE: int = 10
    
    # Enable HTTPS redirect in production
    ENABLE_HTTPS_REDIRECT: bool = IS_PRODUCTION
    
    # Security headers
    SECURITY_HEADERS: dict = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================


class LoggingConfig:
    """Logging configuration."""
    
    # Log level
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Whether to output logs to console
    CONSOLE_LOGGING: bool = True
    
    # Whether to output logs to files
    FILE_LOGGING: bool = True
    
    # Log file format: 'json' for production, 'text' for development
    LOG_FORMAT: Literal["json", "text"] = "json" if IS_PRODUCTION else "text"
    
    # Access log format (for HTTP requests)
    ACCESS_LOG_FORMAT: str = (
        '%(asctime)s - %(client_addr)s - %(request_line)s - '
        '%(status_code)s - %(duration_ms)dms'
    )


# ============================================================================
# CACHING CONFIGURATION
# ============================================================================


class CacheConfig:
    """Configuration for caching strategies."""
    
    # Enable response caching
    ENABLE_CACHING: bool = not IS_DEVELOPMENT
    
    # Cache TTL (time-to-live) in seconds
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # Maximum cache size
    MAX_CACHE_SIZE: int = 1000


# ============================================================================
# MONITORING AND METRICS CONFIGURATION
# ============================================================================


class MetricsConfig:
    """Configuration for application metrics and monitoring."""
    
    # Enable Prometheus metrics
    ENABLE_METRICS: bool = True
    
    # Metrics port
    METRICS_PORT: int = 9090
    
    # Enable performance monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = not IS_DEVELOPMENT


# ============================================================================
# TESTING CONFIGURATION
# ============================================================================


class TestConfig:
    """Configuration for testing."""
    
    # Test database file
    TEST_DATABASE_FILE: Path = BASE_DIR / "test_database.pkl"
    
    # Test upload folder
    TEST_UPLOAD_FOLDER: Path = BASE_DIR / "test_uploads"
    
    # Test timeout in seconds
    TEST_TIMEOUT: int = 30


# ============================================================================
# AGGREGATED CONFIGURATION CLASS
# ============================================================================


class AppConfig:
    """
    Main application configuration class.
    
    Aggregates all configuration sections and provides a single
    point of access to all application settings.
    
    Example:
        >>> config = AppConfig()
        >>> config.server.port
        5000
        >>> config.audio.sample_rate
        22050
    """
    
    # Configuration sections
    paths: PathConfig = PathConfig()
    audio: AudioProcessingConfig = AudioProcessingConfig()
    model: ModelConfig = ModelConfig()
    server: ServerConfig = ServerConfig()
    cors: CorsConfig = CorsConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    cache: CacheConfig = CacheConfig()
    metrics: MetricsConfig = MetricsConfig()
    test: TestConfig = TestConfig()
    
    # Environment info
    environment: Environment = ENVIRONMENT
    is_production: bool = IS_PRODUCTION
    is_development: bool = IS_DEVELOPMENT
    is_staging: bool = IS_STAGING
    
    @classmethod
    def initialize(cls) -> None:
        """
        Initialize application configuration.
        
        Should be called once at application startup.
        Creates necessary directories and validates configuration.
        """
        PathConfig.ensure_directories_exist()


# ============================================================================
# MODULE-LEVEL EXPORTS (for backward compatibility)
# ============================================================================

# Legacy exports for backward compatibility
config: AppConfig = AppConfig()

# Direct access to commonly used values
DATABASE_FILE: Path = PathConfig.DATABASE_FILE
UPLOAD_FOLDER: Path = PathConfig.UPLOAD_FOLDER
DATASET_PATH: Path = PathConfig.DATASET_PATH

SAMPLE_RATE: int = AudioProcessingConfig.SAMPLE_RATE
MAX_DURATION_SECONDS: int = AudioProcessingConfig.MAX_DURATION_SECONDS
WAVEFORM_DISPLAY_SAMPLES: int = AudioProcessingConfig.WAVEFORM_DISPLAY_SAMPLES

KNN_NEIGHBORS: int = ModelConfig.KNN_NEIGHBORS
KNN_METRIC: str = ModelConfig.KNN_METRIC
KNN_WEIGHTS: str = ModelConfig.KNN_WEIGHTS

FEATURE_VECTOR_LENGTH: int = ModelConfig.FEATURE_VECTOR_LENGTH
MFCC_COEFFICIENTS: int = ModelConfig.MFCC_COEFFICIENTS

PORT: int = ServerConfig.PORT
DEBUG: bool = ServerConfig.DEBUG
