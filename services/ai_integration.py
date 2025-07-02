"""
Fixed AI Integration Service for the eBook Editor.
Handles both local Hugging Face models and external AI endpoints.
Provides real text enhancement, suggestions, and generation capabilities.
"""

import asyncio
import json
import time
import re
import threading
from typing import Dict, List, Optional, Any, Union
import httpx
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    pipeline, BartTokenizer, BartForConditionalGeneration, AutoConfig
)
from sentence_transformers import SentenceTransformer
import structlog
import hashlib
import os

from config import Settings, AI_MODEL_CONFIGS

logger = structlog.get_logger()


class AIService:
    """Fixed AI Service for text processing and enhancement."""
    
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
        
        # Suggestion cache to avoid repeated AI calls
        self.suggestion_cache = {}
        self.cache_max_size = 100
        
        # Thread lock for model operations
        self.model_lock = threading.Lock()
        
    async def initialize(self):
        """Initialize AI service with configured models."""
        try:
            logger.info("Initializing AI service", use_local=self.settings.USE_LOCAL_MODEL)
            
            if self.settings.USE_LOCAL_MODEL:
                await self._initialize_local_models()
            
            if self.settings.EXTERNAL_AI_ENABLED:
                await self._initialize_external_client()
            
            # Initialize sentence transformer for semantic analysis
            await self._initialize_sentence_transformer()
            
            self.is_initialized = True
            logger.info("AI service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize AI service", error=str(e))
            # Don't raise in development mode, just log
            if not self.settings.DEBUG:
                raise
    
    async def _initialize_local_models(self):
        """Initialize local Hugging Face models."""
        try:
            # Main text generation model
            model_name = self.settings.LOCAL_MODEL_NAME
            logger.info("Loading main model", model_name=model_name)
            
            with self.model_lock:
                self.local_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.settings.MODEL_CACHE_DIR
                )
                
                # Handle tokenizer padding
                if self.local_tokenizer.pad_token is None:
                    self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
                
                device = self.settings.MODEL_DEVICE
                dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
                
                # Load with appropriate settings
                model_kwargs = {
                    "cache_dir": self.settings.MODEL_CACHE_DIR,
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True
                }
                
                if device == "cuda" and torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"
                
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                
                if device == "cpu" or not torch.cuda.is_available():
                    self.local_model = self.local_model.to("cpu")
                
                self.current_model_name = model_name
            
            # Initialize specialized models
            await self._initialize_specialized_models()
            
            logger.info("Local models loaded successfully")
            
        except Exception as e:
            logger.error("Failed to initialize local models", error=str(e))
            # Create a fallback mock service for development
            self.local_model = None
            self.local_tokenizer = None
            logger.warning("Using fallback mock AI service")
    
    async def _initialize_specialized_models(self):
        """Initialize specialized models for grammar, style, etc."""
        try:
            # Grammar checking model (simplified for now)
            if self.settings.ENABLE_GRAMMAR_CHECK:
                logger.info("Initializing grammar analysis")
                # We'll use rule-based grammar checking for now
                
            # Summarization pipeline for content enhancement
            if self.settings.SUMMARIZATION_MODEL:
                logger.info("Loading summarization pipeline", model=self.settings.SUMMARIZATION_MODEL)
                try:
                    self.summarization_pipeline = pipeline(
                        "summarization",
                        model=self.settings.SUMMARIZATION_MODEL,
                        tokenizer=self.settings.SUMMARIZATION_MODEL,
                        device=0 if self.settings.MODEL_DEVICE == "cuda" and torch.cuda.is_available() else -1,
                        model_kwargs={"cache_dir": self.settings.MODEL_CACHE_DIR}
                    )
                except Exception as e:
                    logger.warning("Failed to load summarization pipeline", error=str(e))
            
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
            if self.settings.MODEL_DEVICE == "cuda" and torch.cuda.is_available():
                self.sentence_transformer = self.sentence_transformer.to('cuda')
        except Exception as e:
            logger.warning("Failed to load sentence transformer", error=str(e))
    
    async def _initialize_external_client(self):
        """Initialize external AI API client."""
        headers = {"Content-Type": "application/json"}
        if self.settings.EXTERNAL_AI_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.EXTERNAL_AI_API_KEY}"
            
        self.external_client = httpx.AsyncClient(
            timeout=self.settings.EXTERNAL_AI_TIMEOUT,
            headers=headers
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
            self.settings.EXTERNAL_AI_ENABLED = (ai_type == "external")
            
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
        """OpenAI-compatible chat completion with proper error handling."""
        try:
            if self.settings.EXTERNAL_AI_ENABLED and self.external_client:
                return await self._external_completion(messages, **kwargs)
            elif self.settings.USE_LOCAL_MODEL and self.local_model:
                return await self._local_completion(messages, **kwargs)
            else:
                # Fallback to mock response for development
                return self._create_mock_completion(messages, **kwargs)
        except Exception as e:
            logger.error("Chat completion failed", error=str(e))
            # Return mock response instead of raising
            return self._create_mock_completion(messages, **kwargs)
    
    async def _local_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Generate completion using local model with proper error handling."""
        if not self.local_model or not self.local_tokenizer:
            logger.warning("Local model not available, using mock response")
            return self._create_mock_completion(messages, **kwargs)
        
        try:
            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages)
            
            with self.model_lock:
                # Tokenize input
                inputs = self.local_tokenizer.encode(
                    prompt, 
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                )
                
                device = next(self.local_model.parameters()).device
                inputs = inputs.to(device)
                
                # Generate response
                max_new_tokens = min(kwargs.get("max_tokens", 256), 512)
                temperature = kwargs.get("temperature", 0.7)
                
                with torch.no_grad():
                    outputs = self.local_model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.local_tokenizer.eos_token_id,
                        attention_mask=torch.ones_like(inputs),
                        repetition_penalty=1.1
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
            return self._create_mock_completion(messages, **kwargs)
    
    async def _external_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Generate completion using external API."""
        if not self.external_client:
            return self._create_mock_completion(messages, **kwargs)
        
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
            return self._create_mock_completion(messages, **kwargs)
    
    def _create_mock_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Create a mock completion response for development/fallback."""
        # Generate a reasonable mock response based on the last message
        last_message = messages[-1]["content"] if messages else ""
        
        mock_response = "I'm analyzing your text. This is a development response. "
        if "grammar" in last_message.lower():
            mock_response += "I've found some potential grammar improvements."
        elif "style" in last_message.lower():
            mock_response += "Here are some style enhancement suggestions."
        else:
            mock_response += "Here's my analysis of your content."
        
        return {
            "id": f"chatcmpl-mock-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": mock_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": len(mock_response.split()),
                "total_tokens": len(last_message.split()) + len(mock_response.split())
            }
        }
    
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
        """Get real AI-powered writing suggestions with caching."""
        try:
            # Create cache key
            cache_key = hashlib.md5(f"{text[:100]}-{str(categories)}".encode()).hexdigest()
            
            # Check cache first
            if cache_key in self.suggestion_cache:
                logger.debug("Returning cached suggestions")
                return self.suggestion_cache[cache_key]
            
            suggestions = []
            
            if not categories:
                categories = ["grammar", "style", "clarity", "structure"]
            
            # Get real suggestions for each category
            for category in categories:
                category_suggestions = await self._get_real_category_suggestions(text, category)
                suggestions.extend(category_suggestions)
            
            # Cache results
            if len(self.suggestion_cache) >= self.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.suggestion_cache))
                del self.suggestion_cache[oldest_key]
            
            self.suggestion_cache[cache_key] = suggestions
            return suggestions
            
        except Exception as e:
            logger.error("Failed to get writing suggestions", error=str(e))
            return self._generate_fallback_suggestions(text, categories or [])
    
    async def _get_real_category_suggestions(self, text: str, category: str) -> List[Dict]:
        """Get real suggestions for a specific category using AI."""
        try:
            prompt = self._create_detailed_suggestion_prompt(text, category)
            
            response = await self.chat_completion([
                {"role": "system", "content": "You are a professional editor specializing in writing improvement. Provide specific, actionable suggestions."},
                {"role": "user", "content": prompt}
            ], max_tokens=512, temperature=0.3)
            
            suggestions_text = response["choices"][0]["message"]["content"]
            
            # Parse the AI response into structured suggestions
            return self._parse_ai_suggestions(suggestions_text, category, text)
            
        except Exception as e:
            logger.error("Failed to get real category suggestions", category=category, error=str(e))
            return self._generate_category_fallback(text, category)
    
    def _create_detailed_suggestion_prompt(self, text: str, category: str) -> str:
        """Create a detailed prompt for getting suggestions in a specific category."""
        # Limit text to avoid token limits
        text_sample = text[:1500] if len(text) > 1500 else text
        
        prompts = {
            "grammar": f"""Please analyze the following text for grammar issues and provide specific corrections:

Text to analyze:
"{text_sample}"

Please respond with specific grammar corrections in this format:
1. Issue: [specific grammatical error]
   Suggestion: [corrected version]
   Explanation: [brief explanation]

Focus on subject-verb agreement, tense consistency, and sentence structure.""",
            
            "style": f"""Please analyze the following text for style improvements:

Text to analyze:
"{text_sample}"

Please provide specific style improvements in this format:
1. Issue: [style issue]
   Suggestion: [improved version]
   Explanation: [why this improves the style]

Focus on word choice, sentence variety, and flow.""",
            
            "clarity": f"""Please analyze the following text for clarity improvements:

Text to analyze:
"{text_sample}"

Please provide specific clarity improvements in this format:
1. Issue: [unclear element]
   Suggestion: [clearer version]
   Explanation: [how this improves clarity]

Focus on ambiguous phrases, complex sentences, and unclear references.""",
            
            "structure": f"""Please analyze the following text for structural improvements:

Text to analyze:
"{text_sample}"

Please provide specific structural improvements in this format:
1. Issue: [structural problem]
   Suggestion: [improved structure]
   Explanation: [how this improves organization]

Focus on paragraph organization, transitions, and logical flow."""
        }
        
        return prompts.get(category, f"Please improve the following text for {category}:\n\n{text_sample}")
    
    def _parse_ai_suggestions(self, suggestions_text: str, category: str, original_text: str) -> List[Dict]:
        """Parse AI-generated suggestions into structured format."""
        try:
            suggestions = []
            lines = suggestions_text.split('\n')
            
            current_suggestion = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                    # Save previous suggestion if complete
                    if current_suggestion.get('issue') and current_suggestion.get('suggestion'):
                        suggestions.append(self._format_suggestion(current_suggestion, category, original_text))
                    current_suggestion = {}
                
                if 'Issue:' in line:
                    current_suggestion['issue'] = line.split('Issue:', 1)[1].strip()
                elif 'Suggestion:' in line:
                    current_suggestion['suggestion'] = line.split('Suggestion:', 1)[1].strip()
                elif 'Explanation:' in line:
                    current_suggestion['explanation'] = line.split('Explanation:', 1)[1].strip()
                elif not current_suggestion.get('issue') and len(line) > 10:
                    # Treat as issue if no explicit format
                    current_suggestion['issue'] = line
            
            # Add last suggestion
            if current_suggestion.get('issue') and current_suggestion.get('suggestion'):
                suggestions.append(self._format_suggestion(current_suggestion, category, original_text))
            
            return suggestions[:3]  # Limit to 3 suggestions per category
            
        except Exception as e:
            logger.error("Failed to parse AI suggestions", error=str(e))
            return self._generate_category_fallback(original_text, category)
    
    def _format_suggestion(self, suggestion_data: Dict, category: str, original_text: str) -> Dict:
        """Format a suggestion into the required structure."""
        issue = suggestion_data.get('issue', '')
        suggested = suggestion_data.get('suggestion', '')
        explanation = suggestion_data.get('explanation', f'AI-suggested improvement for {category}')
        
        # Try to find the position of the issue in the text
        position = original_text.lower().find(issue.lower()[:50]) if issue else 0
        if position == -1:
            position = 0
        
        return {
            "category": category,
            "issue": issue,
            "suggestion": suggested,
            "explanation": explanation,
            "confidence": 0.8,  # High confidence for AI suggestions
            "position": position,
            "length": len(issue)
        }
    
    def _generate_fallback_suggestions(self, text: str, categories: List[str]) -> List[Dict]:
        """Generate intelligent fallback suggestions when AI is not available."""
        suggestions = []
        
        for category in categories:
            suggestions.extend(self._generate_category_fallback(text, category))
        
        return suggestions
    
    def _generate_category_fallback(self, text: str, category: str) -> List[Dict]:
        """Generate fallback suggestions for a specific category using rule-based analysis."""
        suggestions = []
        
        if category == "grammar":
            # Simple grammar checks
            if " there " in text.lower() and " their " in text.lower():
                suggestions.append({
                    "category": "grammar",
                    "issue": "there/their confusion",
                    "suggestion": "Check usage of 'there' vs 'their'",
                    "explanation": "Ensure correct usage of homophones",
                    "confidence": 0.6,
                    "position": text.lower().find(" there "),
                    "length": 5
                })
            
            # Check for passive voice
            passive_indicators = ["was", "were", "been", "is being"]
            for indicator in passive_indicators:
                if f" {indicator} " in text.lower():
                    suggestions.append({
                        "category": "grammar",
                        "issue": f"Possible passive voice: '{indicator}'",
                        "suggestion": "Consider using active voice",
                        "explanation": "Active voice is often clearer and more engaging",
                        "confidence": 0.5,
                        "position": text.lower().find(f" {indicator} "),
                        "length": len(indicator)
                    })
                    break  # Only suggest once per category
        
        elif category == "style":
            # Check for repetitive words
            words = text.lower().split()
            word_counts = {}
            for word in words:
                if len(word) > 4:  # Only check longer words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            for word, count in word_counts.items():
                if count > 3:
                    suggestions.append({
                        "category": "style",
                        "issue": f"Word repetition: '{word}' used {count} times",
                        "suggestion": f"Consider synonyms for '{word}'",
                        "explanation": "Varied vocabulary makes writing more engaging",
                        "confidence": 0.7,
                        "position": text.lower().find(word),
                        "length": len(word)
                    })
                    break  # Only suggest once per category
        
        elif category == "clarity":
            # Check for very long sentences
            sentences = text.split('.')
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) > 30:
                    suggestions.append({
                        "category": "clarity",
                        "issue": "Very long sentence",
                        "suggestion": "Consider breaking into shorter sentences",
                        "explanation": "Shorter sentences improve readability",
                        "confidence": 0.6,
                        "position": text.find(sentence),
                        "length": len(sentence)
                    })
                    break
        
        elif category == "structure":
            # Check paragraph length
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                if len(paragraph.split()) > 150:
                    suggestions.append({
                        "category": "structure",
                        "issue": "Very long paragraph",
                        "suggestion": "Consider breaking into shorter paragraphs",
                        "explanation": "Shorter paragraphs improve readability",
                        "confidence": 0.6,
                        "position": text.find(paragraph),
                        "length": min(50, len(paragraph))
                    })
                    break
        
        return suggestions[:1]  # Return only one suggestion per category for fallback
    
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
            # Return slightly modified text as fallback
            return self._basic_text_enhancement(text)
    
    def _basic_text_enhancement(self, text: str) -> str:
        """Basic text enhancement when AI is not available."""
        # Simple rule-based enhancements
        enhanced = text
        
        # Fix double spaces
        enhanced = re.sub(r'\s+', ' ', enhanced)
        
        # Ensure proper capitalization after periods
        enhanced = re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), enhanced)
        
        # Fix common contractions
        contractions = {
            " dont ": " don't ",
            " cant ": " can't ",
            " wont ": " won't ",
            " its ": " it's ",
        }
        
        for wrong, right in contractions.items():
            enhanced = enhanced.replace(wrong, right)
        
        return enhanced
    
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
            # Basic extractive summary
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    async def detect_chapters(self, text: str) -> List[Dict]:
        """AI-powered chapter detection with fallback."""
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
        
        # Look for chapter markers first
        chapter_pattern = re.compile(r'(Chapter\s+\d+|CHAPTER\s+\d+|\d+\.\s+[A-Z])', re.IGNORECASE)
        matches = list(chapter_pattern.finditer(text))
        
        if len(matches) > 1:
            # Found explicit chapter markers
            for i, match in enumerate(matches):
                start_pos = match.start()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                
                chapter_text = text[start_pos:end_pos]
                title = match.group().strip()
                
                chapters.append({
                    "start_position": start_pos,
                    "title": title,
                    "summary": chapter_text[:100] + "...",
                    "word_count": len(chapter_text.split())
                })
        else:
            # No explicit markers, split by length
            words = text.split()
            chapter_size = max(1000, len(words) // 10)  # Aim for ~10 chapters
            
            for i in range(0, len(words), chapter_size):
                chapter_words = words[i:i + chapter_size]
                start_pos = len(" ".join(words[:i]))
                
                chapters.append({
                    "start_position": start_pos,
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
            elif self.settings.EXTERNAL_AI_ENABLED:
                if self.external_client:
                    # Quick health check to external API
                    try:
                        response = await asyncio.wait_for(
                            self.external_client.get(f"{self.settings.EXTERNAL_AI_BASE_URL}/models"),
                            timeout=5.0
                        )
                        return response.status_code == 200
                    except:
                        return False
                return False
            else:
                # Mock service is always "healthy" for development
                return True
        except:
            return False
    
    async def is_ready(self) -> bool:
        """Check if AI service is ready for processing."""
        return self.is_initialized and await self.is_healthy()
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear caches
            self.model_cache.clear()
            self.suggestion_cache.clear()
            
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