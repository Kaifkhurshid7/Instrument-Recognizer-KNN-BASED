"""
FastAPI application server for instrument recognition.

This module provides the main REST API endpoints for the Instrument Recognizer
application. It includes:
- Audio file upload and processing
- Instrument classification
- Health checks and diagnostics
- Comprehensive error handling and validation
- Structured logging for observability

Endpoints:
    GET  /health              - Server health and model status
    GET  /docs                - Interactive Swagger documentation
    POST /v1/analyze          - Analyze audio and classify instrument

Author: Instrument Recognizer Team
Date: 2024
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZIPMiddleware
from pydantic import ValidationError

from config import (
    ServerConfig,
    CorsConfig,
    PathConfig,
    AppConfig,
    AudioProcessingConfig,
)
from logger import setup_logger
from validators import (
    AudioUploadRequest,
    AnalysisResponse,
    HealthCheckResponse,
    AudioConfig,
    InvalidAudioFormatError,
    AudioFileSizeError,
    AudioDurationError,
    AudioProcessingError,
    FeatureExtractionError,
    ModelNotReadyError,
    ClassificationError,
    InstrumentRecognizerException,
    validate_audio_file,
    get_supported_formats,
)
from classifier import InstrumentClassifier
from feature_extraction import extract_features
from pydub import AudioSegment

# ============================================================================
# LOGGER SETUP
# ============================================================================

logger = setup_logger(__name__)

# ============================================================================
# GLOBAL APPLICATION STATE
# ============================================================================

# Global classifier instance (loaded on startup)
classifier: Optional[InstrumentClassifier] = None


# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    
    Startup:
    - Initialize directories
    - Load and train classifier
    - Log initialization status
    
    Shutdown:
    - Cleanup resources
    - Close database connections
    - Log shutdown status
    """
    # Startup
    try:
        logger.info("=" * 80)
        logger.info(f"Starting Instrument Recognizer API v{ServerConfig.API_VERSION}")
        logger.info(f"Environment: {AppConfig.environment.value}")
        logger.info(f"Log Level: {AppConfig.logging.LOG_LEVEL}")
        logger.info("=" * 80)
        
        # Create necessary directories
        PathConfig.ensure_directories_exist()
        logger.debug("Directories created/verified")
        
        # Initialize classifier
        global classifier
        classifier = InstrumentClassifier()
        classifier.load_and_train()
        logger.info("✓ Classifier loaded and trained successfully")
        
    except Exception as e:
        logger.critical(
            "Failed to initialize application",
            extra={"error": str(e)},
            exc_info=True
        )
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Instrument Recognizer API")
    logger.debug("Cleanup completed")


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================


app = FastAPI(
    title=ServerConfig.API_TITLE,
    description=ServerConfig.API_DESCRIPTION,
    version=ServerConfig.API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CorsConfig.ALLOWED_ORIGINS,
    allow_credentials=CorsConfig.ALLOW_CREDENTIALS,
    allow_methods=CorsConfig.ALLOW_METHODS,
    allow_headers=CorsConfig.ALLOW_HEADERS,
)

# Add GZIP compression middleware for responses
app.add_middleware(GZIPMiddleware, minimum_size=1000)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with detailed messages."""
    logger.warning(
        "validation_error",
        extra={
            "path": str(request.url.path),
            "errors": exc.error_count()
        }
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
        },
    )


@app.exception_handler(InvalidAudioFormatError)
async def invalid_format_handler(request: Request, exc: InvalidAudioFormatError):
    """Handle unsupported audio format errors."""
    logger.warning(f"invalid_audio_format: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid audio format",
            "message": str(exc),
            "supported_formats": get_supported_formats(),
        },
    )


@app.exception_handler(AudioFileSizeError)
async def file_size_handler(request: Request, exc: AudioFileSizeError):
    """Handle file size limit exceeded errors."""
    logger.warning(f"file_size_exceeded: {exc}")
    return JSONResponse(
        status_code=413,
        content={
            "error": "File too large",
            "message": str(exc),
            "max_size_mb": AudioConfig.MAX_FILE_SIZE_BYTES / (1024 * 1024),
        },
    )


@app.exception_handler(AudioDurationError)
async def duration_handler(request: Request, exc: AudioDurationError):
    """Handle audio duration limit exceeded errors."""
    logger.warning(f"audio_duration_exceeded: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Audio duration exceeded",
            "message": str(exc),
            "max_duration_seconds": AudioConfig.MAX_DURATION_SECONDS,
        },
    )


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    """Handle model not initialized errors."""
    logger.error(f"model_not_ready: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Model not ready",
            "message": "Classification model is not initialized. Please try again later.",
        },
    )


@app.exception_handler(AudioProcessingError)
async def audio_processing_handler(request: Request, exc: AudioProcessingError):
    """Handle audio processing errors."""
    logger.warning(f"audio_processing_error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Audio processing failed",
            "message": str(exc),
        },
    )


@app.exception_handler(FeatureExtractionError)
async def feature_extraction_handler(request: Request, exc: FeatureExtractionError):
    """Handle feature extraction errors."""
    logger.warning(f"feature_extraction_error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Feature extraction failed",
            "message": str(exc),
        },
    )


@app.exception_handler(ClassificationError)
async def classification_handler(request: Request, exc: ClassificationError):
    """Handle classification errors."""
    logger.warning(f"classification_error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Classification failed",
            "message": str(exc),
        },
    )


@app.exception_handler(InstrumentRecognizerException)
async def app_exception_handler(request: Request, exc: InstrumentRecognizerException):
    """Handle generic application exceptions."""
    logger.error(f"application_error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"http_error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions."""
    logger.error(
        "unhandled_exception",
        extra={"path": str(request.url.path)},
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please contact support.",
        },
    )


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Diagnostics"],
    summary="Health Check",
    description="Check server status and model readiness",
)
async def health_check() -> HealthCheckResponse:
    """
    Check application health and model status.
    
    Returns:
        HealthCheckResponse with status and model readiness
        
    Response Codes:
        200: Server is operational
        503: Server or model not ready
        
    Example:
        GET /health
        Response:
        {
            "status": "ok",
            "model_ready": true,
            "version": "2.0.0"
        }
    """
    try:
        if classifier is None or not classifier.is_ready:
            logger.warning("Health check: model not ready")
            return HealthCheckResponse(
                status="degraded",
                model_ready=False,
                version=ServerConfig.API_VERSION,
            )
        
        return HealthCheckResponse(
            status="ok",
            model_ready=True,
            version=ServerConfig.API_VERSION,
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthCheckResponse(
            status="error",
            model_ready=False,
            version=ServerConfig.API_VERSION,
        )


# ============================================================================
# ANALYSIS ENDPOINT
# ============================================================================


@app.post(
    "/v1/analyze",
    response_model=AnalysisResponse,
    tags=["Classification"],
    summary="Analyze Audio File",
    description="Upload audio file and get instrument classification with explainability",
    responses={
        200: {"description": "Analysis successful"},
        400: {"description": "Missing or invalid file"},
        413: {"description": "File too large"},
        422: {"description": "Invalid audio format or processing failed"},
        503: {"description": "Model not ready"},
    },
)
async def analyze_audio(
    audioFile: UploadFile = File(..., description="Audio file (MP3, WAV, FLAC, M4A, OGG)")
) -> AnalysisResponse:
    """
    Analyze audio file and classify the instrument.
    
    This endpoint:
    1. Validates the audio file (format, size, duration)
    2. Converts to mono WAV at 22050 Hz
    3. Extracts 26-dimensional spectral features
    4. Classifies using KNN with confidence scores
    5. Returns probabilities and visualization data
    
    Args:
        audioFile: Audio file in supported format
        
    Returns:
        AnalysisResponse with classification result and explainability data
        
    Raises:
        HTTPException: For various error conditions
        
    Example:
        POST /v1/analyze
        Content-Type: multipart/form-data
        
        audioFile: <binary audio data>
        
        Response:
        {
            "instrument": "Piano",
            "confidence_score": 87.34,
            "waveform": {...},
            "feature_vector": [...],
            "compared_vector": [...],
            "knn_probabilities": [...]
        }
    """
    
    # Validate model is ready
    if classifier is None or not classifier.is_ready:
        logger.error("Analysis requested but model not ready")
        raise HTTPException(
            status_code=503,
            detail="Classification model is not initialized"
        )
    
    # Validate file was provided
    if not audioFile or not audioFile.filename:
        logger.warning("Analysis request with no file provided")
        raise HTTPException(
            status_code=400,
            detail="No audio file provided"
        )
    
    request_id = os.urandom(8).hex()
    logger.info(
        "analysis_request_received",
        extra={
            "request_id": request_id,
            "filename": audioFile.filename,
            "content_type": audioFile.content_type,
        }
    )
    
    raw_path = None
    wav_path = None
    
    try:
        # Read file content into memory
        file_content = await audioFile.read()
        file_size = len(file_content)
        
        # Validate file size
        is_valid, error_msg = validate_audio_file(
            audioFile.filename,
            file_size,
            audioFile.content_type,
        )
        
        if not is_valid:
            raise InvalidAudioFormatError(error_msg or "Invalid file")
        
        # Create temporary file paths
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / audioFile.filename
            wav_path = temp_path / "analysis.wav"
            
            # Write uploaded file to temporary location
            with open(raw_path, 'wb') as f:
                f.write(file_content)
            
            logger.debug(
                "file_written_to_temp",
                extra={"request_id": request_id, "file_size_kb": file_size / 1024}
            )
            
            # Convert to mono WAV
            try:
                audio = AudioSegment.from_file(str(raw_path))
                audio = audio.set_channels(1).set_frame_rate(AudioProcessingConfig.SAMPLE_RATE)
                audio.export(str(wav_path), format="wav")
                
                logger.debug(
                    "audio_converted",
                    extra={
                        "request_id": request_id,
                        "duration_ms": len(audio),
                        "frame_rate": audio.frame_rate,
                    }
                )
                
            except Exception as e:
                logger.error(
                    "audio_conversion_failed",
                    extra={"request_id": request_id, "error": str(e)},
                    exc_info=True
                )
                raise AudioProcessingError(
                    f"Failed to convert audio: {e}. "
                    f"Ensure file is a valid audio format."
                )
            
            # Extract features
            try:
                extraction = extract_features(str(wav_path), include_waveform=True)
                
                if extraction.get("features") is None:
                    raise FeatureExtractionError(
                        "Feature extraction returned no data"
                    )
                
                logger.debug(
                    "features_extracted",
                    extra={
                        "request_id": request_id,
                        "feature_dim": extraction["features"].shape[0],
                    }
                )
                
            except (FeatureExtractionError, AudioProcessingError):
                raise
            except Exception as e:
                logger.error(
                    "feature_extraction_error",
                    extra={"request_id": request_id, "error": str(e)},
                    exc_info=True
                )
                raise FeatureExtractionError(f"Feature extraction failed: {e}")
            
            # Classify
            try:
                prediction = classifier.predict(extraction["features"])
                
                logger.info(
                    "analysis_completed",
                    extra={
                        "request_id": request_id,
                        "instrument": prediction.instrument,
                        "confidence": prediction.confidence,
                    }
                )
                
            except (ClassificationError, ModelNotReadyError):
                raise
            except Exception as e:
                logger.error(
                    "classification_error",
                    extra={"request_id": request_id, "error": str(e)},
                    exc_info=True
                )
                raise ClassificationError(f"Classification failed: {e}")
            
            # Build response
            response = AnalysisResponse(
                instrument=prediction.instrument,
                confidence_score=prediction.confidence,
                waveform={
                    "time": extraction.get("time", []),
                    "amplitude": extraction.get("amplitude", []),
                },
                feature_vector=extraction["features"].tolist(),
                compared_vector=prediction.average_vector,
                knn_probabilities=prediction.probabilities,
            )
            
            return response
    
    except (
        InvalidAudioFormatError,
        AudioFileSizeError,
        AudioDurationError,
        AudioProcessingError,
        FeatureExtractionError,
        ClassificationError,
    ):
        # Re-raise our custom exceptions
        raise
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "unexpected_error_in_analysis",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during analysis"
        )


# ============================================================================
# API DOCUMENTATION ENDPOINT
# ============================================================================


@app.get("/", tags=["Documentation"])
async def root():
    """
    Redirect to API documentation.
    
    Returns:
        Redirect to /docs (Swagger UI)
    """
    return {
        "message": "Welcome to Instrument Recognizer API",
        "version": ServerConfig.API_VERSION,
        "documentation": "/docs",
        "health": "/health",
    }


# ============================================================================
# APPLICATION STARTUP
# ============================================================================


def main() -> None:
    """
    Main entry point for the application.
    
    Starts the Uvicorn ASGI server with configured parameters.
    """
    logger.info(f"Starting server on {ServerConfig.HOST}:{ServerConfig.PORT}")
    
    uvicorn.run(
        app,
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        workers=ServerConfig.WORKERS if not ServerConfig.DEBUG else 1,
        reload=ServerConfig.RELOAD,
        log_level=AppConfig.logging.LOG_LEVEL.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
