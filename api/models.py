"""
Pydantic models for API request/response validation.
Defines data structures for the eBook Editor API endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, validator


# Base models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    status: str = "success"
    timestamp: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None


# Text Analysis Models
class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., min_length=1, max_length=1000000)
    include_ai_suggestions: bool = Field(default=True)
    suggestion_categories: Optional[List[str]] = Field(
        default=["grammar", "spelling", "style", "clarity", "structure"]
    )
    language: str = Field(default="en-US")
    readability_metrics: bool = Field(default=True)
    advanced_analysis: bool = Field(default=True)


class GrammarIssue(BaseModel):
    """Grammar issue model."""
    message: str
    offset: int
    length: int
    suggestions: List[str]
    rule_id: str
    category: str
    severity: Literal["error", "warning", "info"] = "warning"


class SpellingError(BaseModel):
    """Spelling error model."""
    word: str
    offset: int
    suggestions: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ReadabilityScores(BaseModel):
    """Readability metrics model."""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    automated_readability_index: float
    coleman_liau_index: float
    grade_level: str
    interpretation: str
    difficulty_level: Literal[
        "very_easy", "easy", "fairly_easy", "standard", 
        "fairly_difficult", "difficult", "very_difficult"
    ]


class StyleSuggestion(BaseModel):
    """Style improvement suggestion."""
    type: Literal["sentence_variety", "word_choice", "clarity", "tone", "structure"]
    message: str
    original_text: str
    suggested_text: Optional[str]
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class AISuggestion(BaseModel):
    """AI-powered writing suggestion."""
    category: Literal["grammar", "spelling", "style", "clarity", "structure", "content"]
    issue: str
    suggestion: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    position: int
    length: int


class TextStatistics(BaseModel):
    """Basic text statistics."""
    character_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    average_words_per_sentence: float
    average_sentences_per_paragraph: float
    lexical_diversity: float
    common_words: List[str]


class StructureAnalysis(BaseModel):
    """Document structure analysis."""
    has_title: bool
    has_headings: bool
    heading_structure: List[Dict[str, Any]]
    paragraph_lengths: List[int]
    sentence_lengths: List[int]
    estimated_reading_time_minutes: float


class TextAnalysisResponse(BaseResponse):
    """Response model for text analysis."""
    statistics: TextStatistics
    grammar_issues: List[GrammarIssue]
    spelling_errors: List[SpellingError]
    readability_scores: ReadabilityScores
    style_suggestions: List[StyleSuggestion]
    ai_suggestions: Optional[List[AISuggestion]] = None
    structure_analysis: StructureAnalysis
    overall_score: float = Field(ge=0.0, le=100.0)
    improvement_areas: List[str]


# AI Configuration Models
class AIConfigRequest(BaseModel):
    """Request model for AI configuration."""
    ai_type: Literal["local", "external"]
    model_name: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_parameters: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = Field(default=30, ge=1, le=300)


class AIConfigResponse(BaseResponse):
    """Response model for AI configuration."""
    ai_type: Literal["local", "external"]
    model_name: str
    is_ready: bool
    capabilities: List[str] = []


# eBook Generation Models
class EBookMetadata(BaseModel):
    """eBook metadata model."""
    title: str = Field(..., min_length=1, max_length=200)
    author: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    publisher: Optional[str] = Field(None, max_length=100)
    language: str = Field(default="en")
    isbn: Optional[str] = None
    genre: Optional[str] = None
    publication_date: Optional[datetime] = None
    copyright: Optional[str] = None
    cover_image_url: Optional[str] = None


class ChapterConfiguration(BaseModel):
    """Chapter detection configuration."""
    detection_method: Literal["ai", "heading", "pagebreak", "manual"] = "ai"
    min_chapter_length: int = Field(default=1000, ge=100)
    max_chapters: int = Field(default=50, ge=1, le=200)
    chapter_title_pattern: Optional[str] = None
    auto_generate_titles: bool = True


class FormatOptions(BaseModel):
    """eBook format options."""
    font_family: str = Field(default="Georgia")
    font_size: int = Field(default=12, ge=8, le=24)
    line_height: float = Field(default=1.5, ge=1.0, le=3.0)
    margin_top: int = Field(default=20, ge=0, le=100)
    margin_bottom: int = Field(default=20, ge=0, le=100)
    margin_left: int = Field(default=20, ge=0, le=100)
    margin_right: int = Field(default=20, ge=0, le=100)
    page_break_before_chapter: bool = True
    include_toc: bool = True
    include_cover: bool = True
    justify_text: bool = True


class AIEnhancementOptions(BaseModel):
    """AI enhancement options for eBook generation."""
    enhance_before_generation: bool = False
    improve_grammar: bool = True
    enhance_style: bool = True
    generate_chapter_summaries: bool = False
    auto_correct_spelling: bool = True
    improve_readability: bool = False
    enhancement_strength: Literal["light", "moderate", "strong"] = "moderate"


class EBookGenerationRequest(BaseModel):
    """Request model for eBook generation."""
    content: str = Field(..., min_length=100)
    format: Literal["epub", "pdf", "docx", "html", "txt"] = "epub"
    metadata: EBookMetadata
    format_options: FormatOptions = Field(default_factory=FormatOptions)
    chapter_config: ChapterConfiguration = Field(default_factory=ChapterConfiguration)
    ai_enhancement_options: AIEnhancementOptions = Field(default_factory=AIEnhancementOptions)
    include_analytics: bool = True


class EBookGenerationResponse(BaseResponse):
    """Response model for eBook generation."""
    file_url: str
    format: str
    file_size_bytes: int
    chapter_count: int
    word_count: int
    estimated_reading_time_minutes: int
    generation_time_seconds: float
    analytics: Optional[Dict[str, Any]] = None


# File Processing Models
class FileUploadResponse(BaseResponse):
    """Response model for file upload."""
    file_id: str
    filename: str
    content_type: str
    file_size_bytes: int
    text_content: str
    word_count: int
    extracted_metadata: Dict[str, Any]
    processing_status: Literal["completed", "processing", "failed"]


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing."""
    file_ids: List[str]
    processing_options: Dict[str, Any] = {}
    output_format: Literal["epub", "pdf", "docx"] = "epub"


class BatchProcessingResponse(BaseResponse):
    """Response model for batch processing."""
    task_id: str
    file_count: int
    estimated_completion_time_minutes: int


# Health and Status Models
class ServiceStatus(BaseModel):
    """Service status model."""
    name: str
    status: Literal["healthy", "degraded", "unhealthy"]
    last_check: datetime
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseResponse):
    """Health check response model."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    services: Dict[str, bool]
    error: Optional[str] = None


# Chat and AI Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request model (OpenAI compatible)."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Chat completion usage model."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response model (OpenAI compatible)."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# Analytics and Reporting Models
class WritingAnalytics(BaseModel):
    """Writing analytics model."""
    session_duration_minutes: float
    words_written: int
    characters_typed: int
    corrections_made: int
    ai_suggestions_accepted: int
    ai_suggestions_rejected: int
    readability_improvement: float
    writing_speed_wpm: float
    most_common_errors: List[str]


class ProjectProgress(BaseModel):
    """Project progress tracking model."""
    project_id: str
    title: str
    target_word_count: int
    current_word_count: int
    completion_percentage: float
    estimated_completion_date: Optional[datetime]
    daily_writing_goals: Dict[str, int]
    writing_streaks: int


class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    api_response_time_ms: float
    ai_processing_time_ms: float
    text_analysis_time_ms: float
    ebook_generation_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float


# Validation helpers
@validator("text", pre=True)
def validate_text_content(cls, v):
    """Validate text content."""
    if not isinstance(v, str):
        raise ValueError("Text must be a string")
    if len(v.strip()) == 0:
        raise ValueError("Text cannot be empty")
    return v.strip()


# Export all models
__all__ = [
    "TextAnalysisRequest", "TextAnalysisResponse",
    "AIConfigRequest", "AIConfigResponse", 
    "EBookGenerationRequest", "EBookGenerationResponse",
    "EBookMetadata", "FormatOptions", "ChapterConfiguration",
    "AIEnhancementOptions", "FileUploadResponse",
    "BatchProcessingRequest", "BatchProcessingResponse",
    "HealthResponse", "ChatCompletionRequest", "ChatCompletionResponse",
    "WritingAnalytics", "ProjectProgress", "PerformanceMetrics"
]
