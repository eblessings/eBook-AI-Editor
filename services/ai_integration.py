"""
AI Integration Service for the eBook Editor.
Handles both local Hugging Face models and external AI endpoints.
Provides text enhancement, suggestions, and generation capabilities.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
import httpx
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    pipeline, BartTokenizer, BartForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import structlog

from config import Settings, AI_MODEL_CONFIGS

logger = structlog.get_logger()


class AIService:
    """AI Service for text processing and enhancement."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.local_model = None
        self.local_tokenizer = None
        self.grammar_model = None
        self.style_model = None
        self.summarization_pipeline = None
        self.sentence_transformer = None
        self.external_client = None
        self.current_model_name = None
        self.is_initialized = False
        
        # Model cache for hot-swapping
        self.model_cache = {}
        self.max_cache_size = 3
        
    async def initialize(self):
        """Initialize AI service with configured models."""
        try:
            logger.info("Initializing AI service", use_local=self.settings.USE_LOCAL_MODEL)
            
            if self.settings.USE_LOCAL_MODEL:
                await self._initialize_local_models()
            else:
                await self._initialize_external_client()
            
            # Initialize sentence transformer for semantic analysis
            await self._initialize_sentence_transformer()
            
            self.is_initialized = True
            logger.info("AI service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize AI service", error=str(e))
            raise
    
    async def _initialize_local_models(self):
        """Initialize local Hugging Face models."""
        try:
            # Main text generation model
            model_name = self.settings.LOCAL_MODEL_NAME
            logger.info("Loading main model", model_name=model_name)
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.settings.MODEL_CACHE_DIR
            )
            
            # Handle tokenizer padding
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            
            device = self.settings.MODEL_DEVICE
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                cache_dir=self.settings.MODEL_CACHE_DIR,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                self.local_model = self.local_model.to(device)
            
            self.current_model_name = model_name
            
            # Initialize specialized models
            await self._initialize_specialized_models()
            
            logger.info("Local models loaded successfully")
            
        except Exception as e:
            logger.error("Failed to initialize local models", error=str(e))
            raise
    
    async def _initialize_specialized_models(self):
        """Initialize specialized models for grammar, style, etc."""
        try:
            # Grammar checking model
            if self.settings.ENABLE_GRAMMAR_CHECK:
                logger.info("Loading grammar model", model=self.settings.GRAMMAR_MODEL)
                self.grammar_model = AutoModelForSequenceClassification.from_pretrained(
                    self.settings.GRAMMAR_MODEL,
                    cache_dir=self.settings.MODEL_CACHE_DIR
                )
            
            # Summarization pipeline for content enhancement
            logger.info("Loading summarization pipeline", model=self.settings.SUMMARIZATION_MODEL)
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.settings.SUMMARIZATION_MODEL,
                tokenizer=self.settings.SUMMARIZATION_MODEL,
                device=0 if self.settings.MODEL_DEVICE == "cuda" else -1,
                model_kwargs={"cache_dir": self.settings.MODEL_CACHE_DIR}
            )
            
        except Exception as e:
            logger.warning("Some specialized models failed to load", error=str(e))
    
    async def _initialize_sentence_transformer(self):
        """Initialize sentence transformer for semantic analysis."""
        try:
            logger.info("Loading sentence transformer")
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=self.settings.MODEL_CACHE_DIR
            )
            if self.settings.MODEL_DEVICE == "cuda":
                self.sentence_transformer = self.sentence_transformer.to('cuda')
        except Exception as e:
            logger.warning("Failed to load sentence transformer", error=str(e))
    
    async def _initialize_external_client(self):
        """Initialize external AI API client."""
        self.external_client = httpx.AsyncClient(
            timeout=self.settings.EXTERNAL_AI_TIMEOUT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.EXTERNAL_AI_API_KEY}"
            } if self.settings.EXTERNAL_AI_API_KEY else {}
        )
        self.current_model_name = self.settings.EXTERNAL_AI_MODEL
        logger.info("External AI client initialized", base_url=self.settings.EXTERNAL_AI_BASE_URL)
    
    async def configure(self, ai_type: str, model_name: Optional[str] = None, 
                       api_endpoint: Optional[str] = None, api_key: Optional[str] = None,
                       model_parameters: Optional[Dict] = None) -> bool:
        """Configure AI service with new settings."""
        try:
            logger.info("Reconfiguring AI service", ai_type=ai_type, model_name=model_name)
            
            # Update settings
            self.settings.USE_LOCAL_MODEL = (ai_type == "local")
            if model_name:
                if ai_type == "local":
                    self.settings.LOCAL_MODEL_NAME = model_name
                else:
                    self.settings.EXTERNAL_AI_MODEL = model_name
            
            if api_endpoint:
                self.settings.EXTERNAL_AI_BASE_URL = api_endpoint
            if api_key:
                self.settings.EXTERNAL_AI_API_KEY = api_key
            
            # Reinitialize with new configuration
            await self.cleanup()
            await self.initialize()
            
            return True
            
        except Exception as e:
            logger.error("Failed to reconfigure AI service", error=str(e))
            return False
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """OpenAI-compatible chat completion."""
        try:
            if self.settings.USE_LOCAL_MODEL:
                return await self._local_completion(messages, **kwargs)
            else:
                return await self._external_completion(messages, **kwargs)
        except Exception as e:
            logger.error("Chat completion failed", error=str(e))
            raise
    
    async def _local_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Generate completion using local model."""
        if not self.local_model or not self.local_tokenizer:
            raise RuntimeError("Local model not initialized")
        
        try:
            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages)
            
            # Tokenize input
            inputs = self.local_tokenizer.encode(
                prompt, 
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            
            if self.settings.MODEL_DEVICE == "cuda":
                inputs = inputs.to('cuda')
            
            # Generate response
            max_new_tokens = kwargs.get("max_tokens", 512)
            temperature = kwargs.get("temperature", 0.7)
            
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            response_text = self.local_tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.current_model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip()
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": inputs.shape[1],
                    "completion_tokens": outputs.shape[1] - inputs.shape[1],
                    "total_tokens": outputs.shape[1]
                }
            }
            
        except Exception as e:
            logger.error("Local completion failed", error=str(e))
            raise
    
    async def _external_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Generate completion using external API."""
        if not self.external_client:
            raise RuntimeError("External client not initialized")
        
        try:
            request_data = {
                "model": self.settings.EXTERNAL_AI_MODEL,
                "messages": messages,
                **kwargs
            }
            
            response = await self.external_client.post(
                f"{self.settings.EXTERNAL_AI_BASE_URL}/chat/completions",
                json=request_data
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error("External completion failed", error=str(e))
            raise
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def get_writing_suggestions(self, text: str, categories: List[str] = None) -> List[Dict]:
        """Get AI-powered writing suggestions."""
        try:
            suggestions = []
            
            if not categories:
                categories = ["grammar", "style", "clarity", "structure"]
            
            for category in categories:
                category_suggestions = await self._get_category_suggestions(text, category)
                suggestions.extend(category_suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error("Failed to get writing suggestions", error=str(e))
            return []
    
    async def _get_category_suggestions(self, text: str, category: str) -> List[Dict]:
        """Get suggestions for a specific category."""
        try:
            if category == "grammar" and self.grammar_model:
                return await self._get_grammar_suggestions(text)
            
            # Use main model for other categories
            prompt = self._create_suggestion_prompt(text, category)
            
            response = await self.chat_completion([
                {"role": "system", "content": "You are a professional editor specializing in writing improvement."},
                {"role": "user", "content": prompt}
            ], max_tokens=512, temperature=0.3)
            
            suggestions_text = response["choices"][0]["message"]["content"]
            
            # Parse suggestions (this would need more sophisticated parsing in production)
            return self._parse_suggestions(suggestions_text, category)
            
        except Exception as e:
            logger.error("Failed to get category suggestions", category=category, error=str(e))
            return []
    
    def _create_suggestion_prompt(self, text: str, category: str) -> str:
        """Create a prompt for getting suggestions in a specific category."""
        prompts = {
            "style": f"""Analyze the following text for style improvements. Provide specific, actionable suggestions for:
1. Sentence variety and flow
2. Word choice optimization
3. Tone consistency
4. Clarity and conciseness

Text: {text[:1000]}...

Respond with a JSON array of suggestions, each with: issue, suggestion, explanation, position.""",
            
            "clarity": f"""Review this text for clarity improvements. Focus on:
1. Unclear or ambiguous sentences
2. Complex sentence structures that could be simplified
3. Jargon or technical terms that need explanation
4. Logical flow and transitions

Text: {text[:1000]}...

Provide specific suggestions as JSON array.""",
            
            "structure": f"""Analyze the structure and organization of this text. Look for:
1. Paragraph organization
2. Logical flow of ideas
3. Transitions between sections
4. Overall document structure

Text: {text[:1000]}...

Suggest structural improvements as JSON array."""
        }
        
        return prompts.get(category, f"Improve the following text for {category}: {text[:1000]}...")
    
    def _parse_suggestions(self, suggestions_text: str, category: str) -> List[Dict]:
        """Parse AI-generated suggestions into structured format."""
        try:
            # Try to parse as JSON first
            if suggestions_text.strip().startswith('['):
                return json.loads(suggestions_text)
            
            # Fallback: parse as text
            suggestions = []
            lines = suggestions_text.split('\n')
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    suggestions.append({
                        "category": category,
                        "issue": "General improvement",
                        "suggestion": line.strip(),
                        "explanation": f"AI-suggested improvement for {category}",
                        "confidence": 0.7,
                        "position": i * 50,
                        "length": 10
                    })
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error("Failed to parse suggestions", error=str(e))
            return []
    
    async def enhance_text(self, text: str, enhancement_type: str = "general") -> str:
        """Enhance text with AI assistance."""
        try:
            prompt = f"""Please enhance the following text for publication quality. Focus on:
1. Grammar and spelling corrections
2. Improved sentence flow and clarity
3. Enhanced vocabulary and style
4. Maintained author's voice and intent

Original text:
{text}

Enhanced version:"""
            
            response = await self.chat_completion([
                {"role": "system", "content": "You are a professional editor. Enhance text while preserving the author's voice."},
                {"role": "user", "content": prompt}
            ], max_tokens=len(text.split()) * 2, temperature=0.3)
            
            enhanced_text = response["choices"][0]["message"]["content"]
            return enhanced_text.strip()
            
        except Exception as e:
            logger.error("Text enhancement failed", error=str(e))
            return text  # Return original if enhancement fails
    
    async def improve_text(self, text: str, improvement_type: str) -> str:
        """Improve text for specific aspects."""
        prompts = {
            "readability": "Make this text more readable and accessible while maintaining its meaning:",
            "formal": "Rewrite this text in a more formal, professional tone:",
            "casual": "Rewrite this text in a more casual, conversational tone:",
            "concise": "Make this text more concise while preserving all important information:",
            "detailed": "Expand this text with more details and explanations:",
            "academic": "Rewrite this text in an academic style with proper citations format:"
        }
        
        prompt = prompts.get(improvement_type, "Improve this text:") + f"\n\n{text}"
        
        try:
            response = await self.chat_completion([
                {"role": "user", "content": prompt}
            ], max_tokens=len(text.split()) * 2, temperature=0.4)
            
            return response["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error("Text improvement failed", improvement_type=improvement_type, error=str(e))
            return text
    
    async def generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate a summary of the text."""
        try:
            if self.summarization_pipeline:
                # Use local summarization model
                summary = self.summarization_pipeline(
                    text,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                return summary[0]['summary_text']
            else:
                # Use main model for summarization
                prompt = f"Summarize the following text in approximately {max_length} words:\n\n{text}"
                response = await self.chat_completion([
                    {"role": "user", "content": prompt}
                ], max_tokens=max_length * 2)
                
                return response["choices"][0]["message"]["content"].strip()
                
        except Exception as e:
            logger.error("Summarization failed", error=str(e))
            return "Summary generation failed"
    
    async def detect_chapters(self, text: str) -> List[Dict]:
        """AI-powered chapter detection."""
        try:
            prompt = f"""Analyze this manuscript and identify natural chapter breaks. 
Look for:
1. Natural story progression points
2. Scene changes
3. Topic shifts
4. Existing chapter markers

For each chapter, provide:
- Start position (character index)
- Suggested title
- Brief summary

Text to analyze:
{text[:5000]}...

Respond with JSON array of chapters."""
            
            response = await self.chat_completion([
                {"role": "system", "content": "You are an expert editor specializing in book structure analysis."},
                {"role": "user", "content": prompt}
            ], max_tokens=1024, temperature=0.3)
            
            chapters_text = response["choices"][0]["message"]["content"]
            
            try:
                return json.loads(chapters_text)
            except:
                # Fallback: simple chapter detection
                return self._simple_chapter_detection(text)
                
        except Exception as e:
            logger.error("Chapter detection failed", error=str(e))
            return self._simple_chapter_detection(text)
    
    def _simple_chapter_detection(self, text: str) -> List[Dict]:
        """Simple fallback chapter detection."""
        chapters = []
        words = text.split()
        chapter_size = max(1000, len(words) // 10)  # Aim for ~10 chapters
        
        for i in range(0, len(words), chapter_size):
            chapter_words = words[i:i + chapter_size]
            chapters.append({
                "start_position": len(" ".join(words[:i])),
                "title": f"Chapter {len(chapters) + 1}",
                "summary": " ".join(chapter_words[:20]) + "...",
                "word_count": len(chapter_words)
            })
        
        return chapters
    
    async def is_healthy(self) -> bool:
        """Check if AI service is healthy."""
        try:
            if self.settings.USE_LOCAL_MODEL:
                return self.local_model is not None and self.local_tokenizer is not None
            else:
                if self.external_client:
                    # Quick health check to external API
                    response = await self.external_client.get(f"{self.settings.EXTERNAL_AI_BASE_URL}/health")
                    return response.status_code == 200
                return False
        except:
            return False
    
    async def is_ready(self) -> bool:
        """Check if AI service is ready for processing."""
        return self.is_initialized and await self.is_healthy()
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear model cache
            self.model_cache.clear()
            
            # Close external client
            if self.external_client:
                await self.external_client.aclose()
                self.external_client = None
            
            # Clear CUDA cache if using GPU
            if self.settings.MODEL_DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("AI service cleanup completed")
            
        except Exception as e:
            logger.error("AI service cleanup failed", error=str(e))
