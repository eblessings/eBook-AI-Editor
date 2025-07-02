"""
Enhanced Main FastAPI application for the eBook Editor.
Fixed multi-worker crashes, improved error handling, and better AI integration.
"""

import asyncio
import logging
import sys
import os
import multiprocessing
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import structlog

from config import settings, AI_MODEL_CONFIGS, EBOOK_FORMATS
from services.ai_integration import AIService
from services.text_processing import TextProcessor
from services.ebook_generator import EBookGenerator
from services.file_handler import FileHandler
from api.models import (
    TextAnalysisRequest, TextAnalysisResponse,
    EBookGenerationRequest, EBookGenerationResponse,
    AIConfigRequest, AIConfigResponse,
    HealthResponse
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global service instances - Use None to handle multi-worker scenarios
ai_service: Optional[AIService] = None
text_processor: Optional[TextProcessor] = None
ebook_generator: Optional[EBookGenerator] = None
file_handler: Optional[FileHandler] = None

# Track if we're in a multi-worker environment
is_main_worker = True


def get_worker_id():
    """Get current worker ID to handle multi-worker scenarios."""
    try:
        import os
        worker_id = os.getpid()
        return worker_id
    except:
        return 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown with multi-worker safety."""
    global ai_service, text_processor, ebook_generator, file_handler, is_main_worker
    
    # Startup
    worker_id = get_worker_id()
    logger.info("Starting eBook Editor Pro", 
                version=settings.APP_VERSION, 
                worker_id=worker_id,
                max_workers=settings.MAX_WORKERS)
    
    # Force single worker if AI is enabled to prevent model sharing issues
    if settings.MAX_WORKERS > 1 and (settings.USE_LOCAL_MODEL or settings.EXTERNAL_AI_ENABLED):
        logger.warning("AI services enabled - forcing single worker mode to prevent crashes")
        is_main_worker = True
    
    try:
        # Initialize services with error handling
        logger.info("Initializing file handler...")
        file_handler = FileHandler(settings)
        
        logger.info("Initializing eBook generator...")
        ebook_generator = EBookGenerator(settings)
        
        # Initialize AI services only in main worker or single worker mode
        if is_main_worker or settings.MAX_WORKERS == 1:
            try:
                logger.info("Initializing text processor...")
                text_processor = TextProcessor(settings)
                await text_processor.initialize()
                
                logger.info("Initializing AI service...")
                ai_service = AIService(settings)
                await ai_service.initialize()
                
            except Exception as e:
                logger.error("Failed to initialize AI services", error=str(e))
                # Continue without AI services in development mode
                if settings.DEBUG:
                    logger.warning("Running without AI services in debug mode")
                    ai_service = None
                    text_processor = None
                else:
                    raise
        else:
            logger.info("Worker process - skipping AI service initialization")
            ai_service = None
            text_processor = None
        
        logger.info("Services initialization completed", 
                   ai_enabled=ai_service is not None,
                   text_processor_enabled=text_processor is not None)
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        # Don't exit in development mode, just log the error
        if not settings.DEBUG:
            sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down eBook Editor Pro", worker_id=worker_id)
    try:
        if ai_service:
            await ai_service.cleanup()
        if text_processor:
            await text_processor.cleanup()
        if file_handler:
            file_handler.cleanup()
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Professional eBook Editor with AI-powered text processing and analysis",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Enhanced dependency functions with better error handling
async def get_ai_service() -> AIService:
    """Get AI service instance with fallback."""
    global ai_service
    if ai_service is None:
        # Try to initialize if not available
        try:
            ai_service = AIService(settings)
            await ai_service.initialize()
            logger.info("AI service initialized on-demand")
        except Exception as e:
            logger.error("Failed to initialize AI service on-demand", error=str(e))
            raise HTTPException(status_code=503, detail="AI service not available")
    return ai_service


async def get_text_processor() -> TextProcessor:
    """Get text processor instance with fallback."""
    global text_processor
    if text_processor is None:
        try:
            text_processor = TextProcessor(settings)
            await text_processor.initialize()
            logger.info("Text processor initialized on-demand")
        except Exception as e:
            logger.error("Failed to initialize text processor on-demand", error=str(e))
            raise HTTPException(status_code=503, detail="Text processor not available")
    return text_processor


async def get_ebook_generator() -> EBookGenerator:
    """Get eBook generator instance."""
    global ebook_generator
    if ebook_generator is None:
        ebook_generator = EBookGenerator(settings)
    return ebook_generator


async def get_file_handler() -> FileHandler:
    """Get file handler instance."""
    global file_handler
    if file_handler is None:
        file_handler = FileHandler(settings)
    return file_handler


# Health check endpoint with comprehensive status
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint."""
    try:
        current_worker = get_worker_id()
        
        # Check if all services are running
        services_status = {
            "ai_service": ai_service is not None and await ai_service.is_healthy() if ai_service else False,
            "text_processor": text_processor is not None and text_processor.is_healthy() if text_processor else False,
            "ebook_generator": ebook_generator is not None,
            "file_handler": file_handler is not None
        }
        
        # Check system resources
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            disk_usage = psutil.disk_usage('.').percent
        except:
            memory_usage = cpu_usage = disk_usage = 0
        
        # Determine overall status
        critical_services = ["file_handler", "ebook_generator"]
        critical_healthy = all(services_status.get(service, False) for service in critical_services)
        ai_healthy = services_status.get("ai_service", False)
        
        if critical_healthy and ai_healthy:
            status = "healthy"
        elif critical_healthy:
            status = "degraded"  # AI not working but core functions available
        else:
            status = "unhealthy"
        
        return HealthResponse(
            status=status,
            version=settings.APP_VERSION,
            services=services_status,
            timestamp=asyncio.get_event_loop().time(),
            uptime_seconds=0,  # Could implement actual uptime tracking
            worker_id=current_worker,
            system_stats={
                "memory_usage_percent": memory_usage,
                "cpu_usage_percent": cpu_usage,
                "disk_usage_percent": disk_usage
            }
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            services={},
            timestamp=asyncio.get_event_loop().time(),
            error=str(e)
        )


# Enhanced file upload with better validation and error handling
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    enhance_with_ai: bool = Form(False),
    target_format: str = Form("epub"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """Enhanced file upload and processing."""
    try:
        logger.info("Processing file upload", 
                   filename=file.filename, 
                   content_type=file.content_type,
                   enhance_with_ai=enhance_with_ai)
        
        # Validate file first
        validation = await file_handler.validate_file(file)
        if not validation['valid']:
            raise HTTPException(status_code=400, detail=validation['error'])
        
        # Reset file position after validation
        await file.seek(0)
        
        # Extract text content
        extracted_data = await file_handler.extract_text_from_file(file)
        
        # Analyze text if text processor is available
        analysis = {}
        try:
            if text_processor:
                analysis = await text_processor.comprehensive_analysis(extracted_data["text"])
            else:
                logger.warning("Text processor not available, skipping analysis")
                analysis = {
                    "statistics": {
                        "word_count": extracted_data.get("word_count", 0),
                        "character_count": extracted_data.get("character_count", 0)
                    }
                }
        except Exception as e:
            logger.warning("Text analysis failed during upload", error=str(e))
            analysis = {}
        
        # Add to background processing if AI enhancement is requested
        if enhance_with_ai and ai_service:
            background_tasks.add_task(
                enhance_text_background,
                extracted_data["text"],
                file.filename
            )
        
        return {
            "file_id": extracted_data["file_id"],
            "filename": file.filename,
            "content_type": file.content_type,
            "text_content": extracted_data["text"],
            "word_count": extracted_data["word_count"],
            "analysis": analysis,
            "processing_status": "enhanced" if not enhance_with_ai else "enhancing",
            "extraction_method": extracted_data.get("extraction_method", "unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File upload failed", 
                    filename=getattr(file, 'filename', 'unknown'), 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


async def enhance_text_background(text: str, filename: str):
    """Background task for AI text enhancement."""
    try:
        if ai_service:
            logger.info("Starting AI text enhancement", filename=filename)
            enhanced_text = await ai_service.enhance_text(text)
            # In a real implementation, you would store this somewhere
            logger.info("AI text enhancement completed", filename=filename)
        else:
            logger.warning("AI service not available for background enhancement")
    except Exception as e:
        logger.error("AI text enhancement failed", filename=filename, error=str(e))


# Enhanced text analysis endpoint
@app.post("/api/analyze")
async def analyze_text(
    request: TextAnalysisRequest,
    text_processor: TextProcessor = Depends(get_text_processor),
    ai_service: AIService = Depends(get_ai_service)
):
    """Enhanced text analysis with better error handling."""
    try:
        if len(request.text) > settings.MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"Text too long. Max length: {settings.MAX_CONTENT_LENGTH} characters"
            )
        
        logger.info("Starting text analysis", 
                   text_length=len(request.text),
                   include_ai=request.include_ai_suggestions)
        
        # Perform comprehensive analysis
        analysis = await text_processor.comprehensive_analysis(request.text)
        
        # Get AI-powered suggestions if requested and available
        if request.include_ai_suggestions and ai_service:
            try:
                suggestions = await ai_service.get_writing_suggestions(
                    request.text,
                    categories=request.suggestion_categories
                )
                analysis["ai_suggestions"] = suggestions
            except Exception as e:
                logger.warning("AI suggestions failed", error=str(e))
                analysis["ai_suggestions"] = []
        elif request.include_ai_suggestions:
            logger.warning("AI suggestions requested but AI service not available")
            analysis["ai_suggestions"] = []
        
        # Ensure required fields exist
        if "statistics" not in analysis:
            analysis["statistics"] = {
                "word_count": len(request.text.split()),
                "character_count": len(request.text)
            }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Enhanced AI configuration endpoint
@app.post("/api/configure-ai")
async def configure_ai(request: AIConfigRequest):
    """Enhanced AI configuration with better validation."""
    try:
        logger.info("Configuring AI settings", 
                   ai_type=request.ai_type,
                   model_name=request.model_name)
        
        global ai_service
        
        # If we have an existing AI service, try to reconfigure it
        if ai_service:
            success = await ai_service.configure(
                ai_type=request.ai_type,
                model_name=request.model_name,
                api_endpoint=request.api_endpoint,
                api_key=request.api_key,
                model_parameters=request.model_parameters
            )
        else:
            # Create new AI service
            try:
                ai_service = AIService(settings)
                # Update settings first
                settings.USE_LOCAL_MODEL = (request.ai_type == "local")
                settings.EXTERNAL_AI_ENABLED = (request.ai_type == "external")
                if request.model_name:
                    if request.ai_type == "local":
                        settings.LOCAL_MODEL_NAME = request.model_name
                    else:
                        settings.EXTERNAL_AI_MODEL = request.model_name
                if request.api_endpoint:
                    settings.EXTERNAL_AI_BASE_URL = request.api_endpoint
                if request.api_key:
                    settings.EXTERNAL_AI_API_KEY = request.api_key
                
                await ai_service.initialize()
                success = True
            except Exception as e:
                logger.error("Failed to create AI service", error=str(e))
                success = False
        
        if not success:
            raise HTTPException(status_code=400, detail="AI configuration failed")
        
        is_ready = await ai_service.is_ready() if ai_service else False
        
        return AIConfigResponse(
            status="success",
            ai_type=request.ai_type,
            model_name=request.model_name or (ai_service.current_model_name if ai_service else "unknown"),
            is_ready=is_ready
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AI configuration failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


# Enhanced eBook generation endpoint
@app.post("/api/generate-ebook")
async def generate_ebook(
    request: EBookGenerationRequest,
    ebook_generator: EBookGenerator = Depends(get_ebook_generator)
):
    """Enhanced eBook generation with better error handling and streaming."""
    try:
        logger.info("Starting eBook generation", 
                   format=request.format, 
                   title=request.metadata.title,
                   content_length=len(request.content))
        
        # Validate inputs
        if not request.content or not request.content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        if not request.metadata.title or not request.metadata.title.strip():
            raise HTTPException(status_code=400, detail="Title is required")
        
        if not request.metadata.author or not request.metadata.author.strip():
            raise HTTPException(status_code=400, detail="Author is required")
        
        # Enhance content with AI if requested
        content = request.content
        if request.ai_enhancement_options.enhance_before_generation and ai_service:
            try:
                content = await ai_service.enhance_text(content)
                logger.info("Content enhanced with AI")
            except Exception as e:
                logger.warning("AI enhancement failed, using original content", error=str(e))
        
        # Generate eBook
        ebook_data = await ebook_generator.create_ebook(
            content=content,
            metadata=request.metadata,
            format_options=request.format_options,
            ai_options=request.ai_enhancement_options
        )
        
        # Prepare filename and response
        safe_title = "".join(c for c in request.metadata.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title.replace(' ', '_')}.{request.format}"
        
        # Get format info
        format_info = EBOOK_FORMATS.get(request.format, {})
        media_type = format_info.get("mime_type", "application/octet-stream")
        
        logger.info("eBook generation completed", 
                   filename=filename,
                   format=request.format)
        
        return StreamingResponse(
            ebook_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": media_type
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("eBook generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"eBook generation failed: {str(e)}")


# Enhanced chat completion endpoint
@app.post("/api/chat/completions")
async def chat_completions(
    request: Dict,
    ai_service: AIService = Depends(get_ai_service)
):
    """Enhanced OpenAI-compatible chat completions endpoint."""
    try:
        logger.debug("Processing chat completion request")
        
        response = await ai_service.chat_completion(
            messages=request.get("messages", []),
            **{k: v for k, v in request.items() if k != "messages"}
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat completion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


# Available models endpoint
@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models."""
    try:
        return {
            "local_models": list(AI_MODEL_CONFIGS.keys()),
            "current_model": ai_service.current_model_name if ai_service else None,
            "supported_formats": list(EBOOK_FORMATS.keys()),
            "ai_service_available": ai_service is not None,
            "text_processor_available": text_processor is not None
        }
    except Exception as e:
        logger.error("Failed to get available models", error=str(e))
        return {
            "local_models": [],
            "current_model": None,
            "supported_formats": list(EBOOK_FORMATS.keys()),
            "ai_service_available": False,
            "text_processor_available": False,
            "error": str(e)
        }


# Text improvement suggestions
@app.post("/api/improve-text")
async def improve_text(
    text: str = Form(...),
    improvement_type: str = Form("general"),
    ai_service: AIService = Depends(get_ai_service)
):
    """Get AI-powered text improvement suggestions."""
    try:
        improved_text = await ai_service.improve_text(text, improvement_type)
        return {"original": text, "improved": improved_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text improvement failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Text improvement failed: {str(e)}")


# Batch processing endpoint
@app.post("/api/batch-process")
async def batch_process_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """Process multiple files in batch."""
    try:
        task_ids = []
        for file in files:
            validation = await file_handler.validate_file(file)
            if validation['valid']:
                task_id = await file_handler.start_batch_processing(file)
                task_ids.append(task_id)
            else:
                logger.warning("File validation failed", 
                             filename=file.filename, 
                             error=validation['error'])
        
        return {"task_ids": task_ids, "status": "processing"}
    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# Static file serving for React build
build_dir = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.exists(build_dir):
    # Serve static files from build directory
    app.mount("/static", StaticFiles(directory=os.path.join(build_dir, "static")), name="static")
    
    # Serve React app for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_react_app(request: Request, full_path: str):
        """Serve React app for all routes that are not API endpoints."""
        # Don't serve React app for API routes, docs, or health checks
        api_paths = ["api/", "docs", "redoc", "health", "static/"]
        if any(full_path.startswith(path) for path in api_paths):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Serve index.html for all other routes (React Router will handle routing)
        return FileResponse(os.path.join(build_dir, "index.html"))
else:
    # Development mode - serve development message
    @app.get("/")
    async def serve_development():
        """Development mode - instructions for building frontend."""
        return {
            "message": "eBook Editor Pro API - Development Mode",
            "status": "frontend_not_built",
            "frontend_status": "Frontend not built. Run build script to build the React app.",
            "api_docs": "/docs",
            "health_check": "/health",
            "worker_info": {
                "worker_id": get_worker_id(),
                "max_workers": settings.MAX_WORKERS,
                "ai_enabled": ai_service is not None
            },
            "build_instructions": {
                "1": "Run: python build.py",
                "2": "Or manually: cd frontend && npm install && npm run build",
                "3": "Then restart the Python server"
            }
        }


# Enhanced error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Enhanced HTTP exception handler."""
    logger.error("HTTP exception", 
                status_code=exc.status_code, 
                detail=exc.detail,
                path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail, 
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Enhanced general exception handler."""
    logger.error("Unhandled exception", 
                error=str(exc), 
                type=type(exc).__name__,
                path=request.url.path)
    
    # Don't expose internal errors in production
    if settings.DEBUG:
        error_detail = str(exc)
    else:
        error_detail = "Internal server error"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": error_detail, 
            "status_code": 500,
            "path": request.url.path
        }
    )


if __name__ == "__main__":
    # Enhanced startup with better configuration
    print(f"""
    🚀 Starting eBook Editor Pro
    
    Configuration:
    - Debug mode: {settings.DEBUG}
    - Host: {settings.HOST}
    - Port: {settings.PORT}
    - Max workers: {settings.MAX_WORKERS}
    - AI enabled: {settings.USE_LOCAL_MODEL or settings.EXTERNAL_AI_ENABLED}
    - Model device: {settings.MODEL_DEVICE}
    
    """)
    
    # Determine optimal worker count
    workers = 1 if settings.DEBUG else settings.MAX_WORKERS
    
    # Force single worker if AI is enabled
    if (settings.USE_LOCAL_MODEL or settings.EXTERNAL_AI_ENABLED) and workers > 1:
        print("⚠️  AI services enabled - forcing single worker to prevent crashes")
        workers = 1
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG and workers == 1,  # Only reload in debug mode with single worker
        log_level=settings.LOG_LEVEL.lower(),
        workers=workers,
        access_log=True
    )