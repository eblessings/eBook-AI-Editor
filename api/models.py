"""
Enhanced Pydantic models for API request/response validation.
Includes fixes for all response types and new AI configuration models.
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


# Enhanced Health Check Models
class SystemStats(BaseModel):
    """System statistics model."""
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0


class HealthResponse(BaseResponse):
    """Enhanced health check response model."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    services: Dict[str, bool]
    timestamp: float
    uptime_seconds: float = 0.0
    worker_id: Optional[int] = None
    system_stats: Optional[SystemStats] = None
    error: Optional[str] = None


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
    reading_time_minutes: float = 0.0


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
    average_words_per_sentence: float = 0.0
    average_sentences_per_paragraph: float = 0.0
    lexical_diversity: float = 0.0
    common_words: List[str] = []
    complex_word_ratio: float = 0.0
    unique_word_count: int = 0


class StructureAnalysis(BaseModel):
    """Document structure analysis."""
    has_title: bool
    has_headings: bool
    heading_structure: List[Dict[str, Any]] = []
    paragraph_count: int = 0
    paragraph_lengths: List[int] = []
    sentence_count: int = 0
    sentence_lengths: List[int] = []
    average_paragraph_length: float = 0.0
    average_sentence_length: float = 0.0
    estimated_reading_time_minutes: float = 0.0


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""
    overall_sentiment: Literal["positive", "negative", "neutral"]
    overall_score: float = 0.0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    neutral_ratio: float = 0.0


class KeywordAnalysis(BaseModel):
    """Keyword analysis results."""
    top_keywords: List[List[Any]] = []  # List of [word, count] pairs
    keyword_density: float = 0.0
    named_entities: Dict[str, int] = {}
    unique_keywords: int = 0
    total_keywords: int = 0


class LinguisticFeatures(BaseModel):
    """Linguistic features analysis."""
    pos_distribution: Dict[str, int] = {}
    dependency_distribution: Dict[str, int] = {}
    subordination_ratio: float = 0.0
    average_dependency_distance: float = 0.0
    linguistic_complexity_score: float = 0.0


class TextAnalysisResponse(BaseResponse):
    """Enhanced response model for text analysis."""
    statistics: TextStatistics
    grammar_issues: List[GrammarIssue] = []
    spelling_errors: List[SpellingError] = []
    readability_scores: ReadabilityScores
    style_suggestions: List[StyleSuggestion] = []
    ai_suggestions: Optional[List[AISuggestion]] = []
    structure_analysis: StructureAnalysis
    sentiment_analysis: Optional[SentimentAnalysis] = None
    keyword_analysis: Optional[KeywordAnalysis] = None
    linguistic_features: Optional[LinguisticFeatures] = None
    overall_score: float = Field(ge=0.0, le=100.0, default=50.0)
    improvement_areas: List[str] = []


# Enhanced AI Configuration Models
class AIConfigRequest(BaseModel):
    """Enhanced request model for AI configuration."""
    ai_type: Literal["local", "external", "claude", "mistral", "openai"]
    model_name: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_parameters: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = Field(default=30, ge=1, le=300)


class AIConfigResponse(BaseResponse):
    """Enhanced response model for AI configuration."""
    ai_type: Literal["local", "external", "claude", "mistral", "openai"]
    model_name: str
    is_ready: bool
    capabilities: List[str] = []
    endpoint: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None


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
    format: Literal["epub", "pdf", "docx", "html", "txt"] = "epub"
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
class FileValidationResult(BaseModel):
    """File validation result."""
    valid: bool
    error: Optional[str] = None
    file_size: Optional[int] = None
    detected_type: Optional[str] = None
    original_type: Optional[str] = None


class ExtractedContent(BaseModel):
    """Extracted content from file."""
    text: str
    metadata: Dict[str, Any] = {}
    structure: Dict[str, Any] = {}
    images: List[Dict[str, Any]] = []
    extraction_method: str = "unknown"


class FileUploadResponse(BaseResponse):
    """Enhanced response model for file upload."""
    file_id: str
    filename: str
    content_type: str
    file_size: int = Field(alias="file_size_bytes", default=0)
    text_content: str
    word_count: int
    character_count: int = 0
    extracted_metadata: Dict[str, Any] = Field(alias="analysis", default={})
    processing_status: Literal["completed", "processing", "failed", "enhanced", "enhancing"]
    extraction_method: str = "unknown"


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


# Chat and AI Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request model (OpenAI compatible)."""
    model: str = "default"
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
    most_common_errors: List[str] = []


class ProjectProgress(BaseModel):
    """Project progress tracking model."""
    project_id: str
    title: str
    target_word_count: int
    current_word_count: int
    completion_percentage: float
    estimated_completion_date: Optional[datetime]
    daily_writing_goals: Dict[str, int] = {}
    writing_streaks: int = 0


class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    api_response_time_ms: float
    ai_processing_time_ms: float
    text_analysis_time_ms: float
    ebook_generation_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float


# Enhanced Available Models Response
class AvailableModelsResponse(BaseModel):
    """Response model for available models endpoint."""
    local_models: List[str] = []
    current_model: Optional[str] = None
    supported_formats: List[str] = []
    ai_service_available: bool = False
    text_processor_available: bool = False
    error: Optional[str] = None


# Text Improvement Models
class TextImprovementRequest(BaseModel):
    """Request model for text improvement."""
    text: str = Field(..., min_length=1, max_length=10000)
    improvement_type: Literal["readability", "formal", "casual", "concise", "detailed", "academic", "general"] = "general"


class TextImprovementResponse(BaseModel):
    """Response model for text improvement."""
    original: str
    improved: str
    improvement_type: str
    changes_made: List[str] = []


# Enhanced validation helpers
@validator("text", pre=True, allow_reuse=True)
def validate_text_content(cls, v):
    """Validate text content."""
    if not isinstance(v, str):
        raise ValueError("Text must be a string")
    if len(v.strip()) == 0:
        raise ValueError("Text cannot be empty")
    return v.strip()


# Custom validators
@validator("confidence", allow_reuse=True)
def validate_confidence(cls, v):
    """Validate confidence score."""
    if not isinstance(v, (int, float)):
        raise ValueError("Confidence must be a number")
    if not 0.0 <= v <= 1.0:
        raise ValueError("Confidence must be between 0 and 1")
    return float(v)


# Export all models
__all__ = [
    # Base models
    "BaseResponse",
    
    # Health check models
    "SystemStats", "HealthResponse",
    
    # Text analysis models
    "TextAnalysisRequest", "TextAnalysisResponse",
    "GrammarIssue", "SpellingError", "ReadabilityScores",
    "StyleSuggestion", "AISuggestion", "TextStatistics",
    "StructureAnalysis", "SentimentAnalysis", "KeywordAnalysis",
    "LinguisticFeatures",
    
    # AI configuration models
    "AIConfigRequest", "AIConfigResponse",
    
    # eBook generation models
    "EBookGenerationRequest", "EBookGenerationResponse",
    "EBookMetadata", "FormatOptions", "ChapterConfiguration",
    "AIEnhancementOptions",
    
    # File processing models
    "FileValidationResult", "ExtractedContent", "FileUploadResponse",
    "BatchProcessingRequest", "BatchProcessingResponse",
    
    # Chat models
    "ChatMessage", "ChatCompletionRequest", "ChatCompletionResponse",
    "ChatCompletionChoice", "ChatCompletionUsage",
    
    # Analytics models
    "WritingAnalytics", "ProjectProgress", "PerformanceMetrics",
    
    # Other models
    "AvailableModelsResponse", "TextImprovementRequest", "TextImprovementResponse"
]