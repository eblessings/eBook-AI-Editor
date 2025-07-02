"""
Main FastAPI application for the Enhanced eBook Editor.
Provides REST API endpoints for text processing, AI integration, and eBook generation.
"""

import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
import argparse
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

# Global service instances
ai_service: Optional[AIService] = None
text_processor: Optional[TextProcessor] = None
ebook_generator: Optional[EBookGenerator] = None
file_handler: Optional[FileHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("Starting eBook Editor Pro", version=settings.APP_VERSION)
    
    global ai_service, text_processor, ebook_generator, file_handler
    
    try:
        # Initialize services
        logger.info("Initializing AI service...")
        ai_service = AIService(settings)
        await ai_service.initialize()
        
        logger.info("Initializing text processor...")
        text_processor = TextProcessor(settings)
        await text_processor.initialize()
        
        logger.info("Initializing eBook generator...")
        ebook_generator = EBookGenerator(settings)
        
        logger.info("Initializing file handler...")
        file_handler = FileHandler(settings)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        # Don't exit in development mode
        if not settings.DEBUG:
            sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down eBook Editor Pro")
    if ai_service:
        await ai_service.cleanup()
    if text_processor:
        await text_processor.cleanup()


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

# Mount API routes first (higher priority)
# All API routes are under /api/ or /docs, /redoc, /health

# Dependency to get services
async def get_ai_service() -> AIService:
    """Get AI service instance."""
    if ai_service is None:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    return ai_service


async def get_text_processor() -> TextProcessor:
    """Get text processor instance."""
    if text_processor is None:
        raise HTTPException(status_code=503, detail="Text processor not initialized")
    return text_processor


async def get_ebook_generator() -> EBookGenerator:
    """Get eBook generator instance."""
    if ebook_generator is None:
        raise HTTPException(status_code=503, detail="eBook generator not initialized")
    return ebook_generator


async def get_file_handler() -> FileHandler:
    """Get file handler instance."""
    if file_handler is None:
        raise HTTPException(status_code=503, detail="File handler not initialized")
    return file_handler


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if all services are running
        services_status = {
            "ai_service": ai_service is not None and await ai_service.is_healthy(),
            "text_processor": text_processor is not None and text_processor.is_healthy(),
            "ebook_generator": ebook_generator is not None,
            "file_handler": file_handler is not None
        }
        
        all_healthy = all(services_status.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version=settings.APP_VERSION,
            services=services_status,
            timestamp=asyncio.get_event_loop().time()
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


# File upload and processing
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    enhance_with_ai: bool = Form(False),
    target_format: str = Form("epub"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file_handler: FileHandler = Depends(get_file_handler),
    text_processor: TextProcessor = Depends(get_text_processor)
):
    """Upload and process manuscript file."""
    try:
        # Validate file
        if file.content_type not in settings.SUPPORTED_UPLOAD_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Process file
        logger.info("Processing uploaded file", filename=file.filename, content_type=file.content_type)
        
        # Extract text content
        extracted_data = await file_handler.extract_text_from_file(file)
        
        # Analyze text
        analysis = await text_processor.comprehensive_analysis(extracted_data["text"])
        
        # Add to background processing if AI enhancement is requested
        if enhance_with_ai:
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
            "processing_status": "enhanced" if not enhance_with_ai else "enhancing"
        }
        
    except Exception as e:
        logger.error("File upload failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


async def enhance_text_background(text: str, filename: str):
    """Background task for AI text enhancement."""
    try:
        logger.info("Starting AI text enhancement", filename=filename)
        enhanced_text = await ai_service.enhance_text(text)
        # Store enhanced text (implementation depends on your storage strategy)
        logger.info("AI text enhancement completed", filename=filename)
    except Exception as e:
        logger.error("AI text enhancement failed", filename=filename, error=str(e))


# Text analysis endpoint
@app.post("/api/analyze", response_model=TextAnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    text_processor: TextProcessor = Depends(get_text_processor),
    ai_service: AIService = Depends(get_ai_service)
):
    """Comprehensive text analysis with AI-powered suggestions."""
    try:
        if len(request.text) > settings.MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"Text too long. Max length: {settings.MAX_CONTENT_LENGTH} characters"
            )
        
        logger.info("Starting text analysis", text_length=len(request.text))
        
        # Perform comprehensive analysis
        analysis = await text_processor.comprehensive_analysis(request.text)
        
        # Get AI-powered suggestions if requested
        if request.include_ai_suggestions:
            suggestions = await ai_service.get_writing_suggestions(
                request.text,
                categories=request.suggestion_categories
            )
            analysis["ai_suggestions"] = suggestions
        
        return TextAnalysisResponse(**analysis)
        
    except Exception as e:
        logger.error("Text analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# AI configuration endpoint
@app.post("/api/configure-ai", response_model=AIConfigResponse)
async def configure_ai(
    request: AIConfigRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Configure AI model settings."""
    try:
        logger.info("Configuring AI settings", ai_type=request.ai_type)
        
        success = await ai_service.configure(
            ai_type=request.ai_type,
            model_name=request.model_name,
            api_endpoint=request.api_endpoint,
            api_key=request.api_key,
            model_parameters=request.model_parameters
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="AI configuration failed")
        
        return AIConfigResponse(
            status="success",
            ai_type=request.ai_type,
            model_name=request.model_name or ai_service.current_model_name,
            is_ready=await ai_service.is_ready()
        )
        
    except Exception as e:
        logger.error("AI configuration failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


# eBook generation endpoint
@app.post("/api/generate-ebook")
async def generate_ebook(
    request: EBookGenerationRequest,
    ebook_generator: EBookGenerator = Depends(get_ebook_generator),
    ai_service: AIService = Depends(get_ai_service)
):
    """Generate eBook from processed content."""
    try:
        logger.info("Starting eBook generation", format=request.format, title=request.metadata.title)
        
        # Enhance content with AI if requested
        content = request.content
        if request.ai_enhancement_options.enhance_before_generation:
            content = await ai_service.enhance_text(content)
        
        # Generate eBook
        ebook_data = await ebook_generator.create_ebook(
            content=content,
            metadata=request.metadata,
            format_options=request.format_options,
            ai_options=request.ai_enhancement_options
        )
        
        # Return as streaming response
        filename = f"{request.metadata.title.replace(' ', '_')}.{request.format}"
        media_type = EBOOK_FORMATS[request.format]["mime_type"]
        
        return StreamingResponse(
            ebook_data,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error("eBook generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"eBook generation failed: {str(e)}")


# AI chat completion endpoint (OpenAI compatible)
@app.post("/api/chat/completions")
async def chat_completions(
    request: Dict,
    ai_service: AIService = Depends(get_ai_service)
):
    """OpenAI-compatible chat completions endpoint."""
    try:
        response = await ai_service.chat_completion(
            messages=request.get("messages", []),
            **{k: v for k, v in request.items() if k != "messages"}
        )
        return response
    except Exception as e:
        logger.error("Chat completion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


# Available models endpoint
@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models."""
    return {
        "local_models": list(AI_MODEL_CONFIGS.keys()),
        "current_model": ai_service.current_model_name if ai_service else None,
        "supported_formats": list(EBOOK_FORMATS.keys())
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
            if file.content_type in settings.SUPPORTED_UPLOAD_TYPES:
                task_id = await file_handler.start_batch_processing(file)
                task_ids.append(task_id)
        
        return {"task_ids": task_ids, "status": "processing"}
    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# Static file serving for React build
# Check if build directory exists
build_dir = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.exists(build_dir):
    # Serve static files from build directory
    app.mount("/static", StaticFiles(directory=os.path.join(build_dir, "static")), name="static")
    
    # Serve React app for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_react_app(request: Request, full_path: str):
        """Serve React app for all routes that are not API endpoints."""
        # Don't serve React app for API routes
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("redoc") or full_path == "health":
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
            "frontend_status": "Frontend not built. Run 'cd frontend && npm run build' to build the React app.",
            "api_docs": "/docs",
            "health_check": "/health",
            "instructions": {
                "1": "Install Node.js and npm",
                "2": "cd frontend",
                "3": "npm install",
                "4": "npm run build",
                "5": "Restart the Python server"
            }
        }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error("HTTP exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), type=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1 if settings.DEBUG else settings.MAX_WORKERS
    )
