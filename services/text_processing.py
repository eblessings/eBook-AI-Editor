"""
Advanced Text Processing Service for the eBook Editor.
Integrates spaCy, NLTK, textstat, and language-tool-python for comprehensive text analysis.
"""

import asyncio
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple
import spacy
import nltk
import textstat
import language_tool_python
from spellchecker import SpellChecker
from textstat import flesch_reading_ease, flesch_kincaid_grade
import structlog

from config import Settings, READABILITY_LEVELS

logger = structlog.get_logger()


class TextProcessor:
    """Advanced text processor with multiple NLP libraries."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.nlp = None
        self.grammar_tool = None
        self.spell_checker = None
        self.is_initialized = False
        
        # Cache for expensive operations
        self.analysis_cache = {}
        self.cache_max_size = 100
    
    async def initialize(self):
        """Initialize all text processing components."""
        try:
            logger.info("Initializing text processor")
            
            # Initialize spaCy
            await self._initialize_spacy()
            
            # Initialize grammar checker
            if self.settings.ENABLE_GRAMMAR_CHECK:
                await self._initialize_grammar_checker()
            
            # Initialize spell checker
            await self._initialize_spell_checker()
            
            # Download required NLTK data
            await self._download_nltk_data()
            
            self.is_initialized = True
            logger.info("Text processor initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize text processor", error=str(e))
            raise
    
    async def _initialize_spacy(self):
        """Initialize spaCy with the configured model."""
        try:
            logger.info("Loading spaCy model", model=self.settings.SPACY_MODEL)
            
            # Try to load the model, download if not available
            try:
                self.nlp = spacy.load(self.settings.SPACY_MODEL)
            except OSError:
                logger.warning("spaCy model not found, using fallback")
                # Fallback to a smaller model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # If no model available, create blank
                    self.nlp = spacy.blank("en")
                    logger.warning("Using blank spaCy model - some features will be limited")
            
            # Configure pipeline
            if self.nlp.has_pipe("ner"):
                self.nlp.disable_pipes(["ner"])  # Disable NER for performance
            
            logger.info("spaCy model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to initialize spaCy", error=str(e))
            raise
    
    async def _initialize_grammar_checker(self):
        """Initialize LanguageTool grammar checker."""
        try:
            logger.info("Initializing grammar checker", language=self.settings.LANGUAGE_TOOL_LANGUAGE)
            
            # Initialize in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.grammar_tool = await loop.run_in_executor(
                None, 
                lambda: language_tool_python.LanguageTool(self.settings.LANGUAGE_TOOL_LANGUAGE)
            )
            
            logger.info("Grammar checker initialized")
            
        except Exception as e:
            logger.error("Failed to initialize grammar checker", error=str(e))
            self.grammar_tool = None
    
    async def _initialize_spell_checker(self):
        """Initialize spell checker."""
        try:
            logger.info("Initializing spell checker")
            self.spell_checker = SpellChecker()
            logger.info("Spell checker initialized")
            
        except Exception as e:
            logger.error("Failed to initialize spell checker", error=str(e))
            self.spell_checker = None
    
    async def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            logger.info("Downloading NLTK data")
            
            # Download required datasets
            nltk_datasets = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'vader_lexicon', 'omw-1.4'
            ]
            
            for dataset in nltk_datasets:
                try:
                    nltk.data.find(f'tokenizers/{dataset}')
                except LookupError:
                    try:
                        nltk.download(dataset, quiet=True)
                    except Exception as e:
                        logger.warning("Failed to download NLTK dataset", dataset=dataset, error=str(e))
            
            logger.info("NLTK data download completed")
            
        except Exception as e:
            logger.error("NLTK data download failed", error=str(e))
    
    async def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis."""
        try:
            if not self.is_initialized:
                raise RuntimeError("Text processor not initialized")
            
            # Check cache first
            text_hash = hash(text)
            if text_hash in self.analysis_cache:
                logger.debug("Returning cached analysis")
                return self.analysis_cache[text_hash]
            
            logger.info("Starting comprehensive text analysis", text_length=len(text))
            
            # Run all analyses
            results = {
                "statistics": await self._calculate_basic_statistics(text),
                "grammar_issues": await self._check_grammar(text),
                "spelling_errors": await self._check_spelling(text),
                "readability_scores": await self._calculate_readability(text),
                "style_suggestions": await self._analyze_style(text),
                "structure_analysis": await self._analyze_structure(text),
                "sentiment_analysis": await self._analyze_sentiment(text),
                "keyword_analysis": await self._analyze_keywords(text),
                "linguistic_features": await self._analyze_linguistic_features(text)
            }
            
            # Calculate overall score
            results["overall_score"] = self._calculate_overall_score(results)
            results["improvement_areas"] = self._identify_improvement_areas(results)
            
            # Cache results
            if len(self.analysis_cache) >= self.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.analysis_cache))
                del self.analysis_cache[oldest_key]
            
            self.analysis_cache[text_hash] = results
            
            logger.info("Text analysis completed")
            return results
            
        except Exception as e:
            logger.error("Comprehensive analysis failed", error=str(e))
            raise
    
    async def _calculate_basic_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics."""
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Basic counts
            sentences = list(doc.sents)
            words = [token for token in doc if not token.is_space and not token.is_punct]
            paragraphs = text.split('\n\n')
            
            # Calculate statistics
            character_count = len(text)
            word_count = len(words)
            sentence_count = len(sentences)
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            # Advanced statistics
            avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
            avg_sentences_per_paragraph = sentence_count / paragraph_count if paragraph_count > 0 else 0
            
            # Lexical diversity (Type-Token Ratio)
            unique_words = set(token.lemma_.lower() for token in words if token.is_alpha)
            lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0
            
            # Most common words
            word_freq = Counter(token.lemma_.lower() for token in words if token.is_alpha and not token.is_stop)
            common_words = [word for word, _ in word_freq.most_common(10)]
            
            # Complexity metrics
            complex_words = [token.text for token in words if len(token.text) > 6]
            complex_word_ratio = len(complex_words) / word_count if word_count > 0 else 0
            
            return {
                "character_count": character_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "average_words_per_sentence": round(avg_words_per_sentence, 2),
                "average_sentences_per_paragraph": round(avg_sentences_per_paragraph, 2),
                "lexical_diversity": round(lexical_diversity, 3),
                "common_words": common_words,
                "complex_word_ratio": round(complex_word_ratio, 3),
                "unique_word_count": len(unique_words)
            }
            
        except Exception as e:
            logger.error("Failed to calculate basic statistics", error=str(e))
            return {}
    
    async def _check_grammar(self, text: str) -> List[Dict[str, Any]]:
        """Check grammar using LanguageTool."""
        if not self.grammar_tool:
            return []
        
        try:
            logger.debug("Checking grammar")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            matches = await loop.run_in_executor(
                None, 
                self.grammar_tool.check, 
                text
            )
            
            grammar_issues = []
            for match in matches[:20]:  # Limit to 20 issues
                # Determine severity
                severity = "error" if "GRAMMAR" in match.ruleId else "warning"
                if "STYLE" in match.ruleId or "TYPOGRAPHY" in match.ruleId:
                    severity = "info"
                
                grammar_issues.append({
                    "message": match.message,
                    "offset": match.offset,
                    "length": match.errorLength,
                    "suggestions": match.replacements[:3],  # Top 3 suggestions
                    "rule_id": match.ruleId,
                    "category": match.category,
                    "severity": severity,
                    "context": self._get_context(text, match.offset, match.errorLength)
                })
            
            logger.debug("Grammar check completed", issues_found=len(grammar_issues))
            return grammar_issues
            
        except Exception as e:
            logger.error("Grammar check failed", error=str(e))
            return []
    
    async def _check_spelling(self, text: str) -> List[Dict[str, Any]]:
        """Check spelling using pyspellchecker."""
        if not self.spell_checker:
            return []
        
        try:
            logger.debug("Checking spelling")
            
            # Extract words and their positions
            word_pattern = re.compile(r'\b[a-zA-Z]+\b')
            words_with_positions = [(match.group(), match.start()) for match in word_pattern.finditer(text)]
            
            # Check spelling
            words = [word for word, _ in words_with_positions]
            misspelled = self.spell_checker.unknown(words)
            
            spelling_errors = []
            for word, position in words_with_positions:
                if word in misspelled:
                    candidates = list(self.spell_checker.candidates(word))
                    suggestions = candidates[:5] if candidates else []
                    
                    # Calculate confidence based on edit distance
                    confidence = 0.8 if suggestions else 0.3
                    
                    spelling_errors.append({
                        "word": word,
                        "offset": position,
                        "suggestions": suggestions,
                        "confidence": confidence,
                        "context": self._get_context(text, position, len(word))
                    })
            
            logger.debug("Spelling check completed", errors_found=len(spelling_errors))
            return spelling_errors[:15]  # Limit to 15 errors
            
        except Exception as e:
            logger.error("Spelling check failed", error=str(e))
            return []
    
    async def _calculate_readability(self, text: str) -> Dict[str, Any]:
        """Calculate readability metrics using textstat."""
        try:
            logger.debug("Calculating readability metrics")
            
            # Calculate various readability scores
            flesch_ease = textstat.flesch_reading_ease(text)
            flesch_grade = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            smog = textstat.smog_index(text)
            ari = textstat.automated_readability_index(text)
            coleman_liau = textstat.coleman_liau_index(text)
            grade_level = textstat.text_standard(text)
            
            # Determine difficulty level
            difficulty_level = self._determine_difficulty_level(flesch_ease)
            
            # Get interpretation
            interpretation = self._interpret_readability_score(flesch_ease)
            
            return {
                "flesch_reading_ease": round(flesch_ease, 2),
                "flesch_kincaid_grade": round(flesch_grade, 2),
                "gunning_fog": round(gunning_fog, 2),
                "smog_index": round(smog, 2),
                "automated_readability_index": round(ari, 2),
                "coleman_liau_index": round(coleman_liau, 2),
                "grade_level": grade_level,
                "difficulty_level": difficulty_level,
                "interpretation": interpretation,
                "reading_time_minutes": self._estimate_reading_time(text)
            }
            
        except Exception as e:
            logger.error("Readability calculation failed", error=str(e))
            return {}
    
    async def _analyze_style(self, text: str) -> List[Dict[str, Any]]:
        """Analyze writing style and provide suggestions."""
        try:
            logger.debug("Analyzing writing style")
            
            doc = self.nlp(text)
            suggestions = []
            
            # Analyze sentence variety
            sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
            if sentence_lengths:
                avg_length = sum(sentence_lengths) / len(sentence_lengths)
                length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
                
                if length_variance < 10:
                    suggestions.append({
                        "type": "sentence_variety",
                        "message": "Consider varying sentence lengths for better flow",
                        "original_text": "",
                        "suggested_text": None,
                        "explanation": "Your sentences have similar lengths. Mix short and long sentences for better rhythm.",
                        "confidence": 0.7
                    })
            
            # Check for passive voice
            passive_sentences = self._detect_passive_voice(doc)
            if len(passive_sentences) > len(list(doc.sents)) * 0.3:
                suggestions.append({
                    "type": "word_choice",
                    "message": "Reduce passive voice usage",
                    "original_text": "",
                    "suggested_text": None,
                    "explanation": "Consider using active voice for more engaging writing.",
                    "confidence": 0.8
                })
            
            # Check for word repetition
            word_repetition = self._check_word_repetition(doc)
            for word, count in word_repetition:
                if count > 5:
                    suggestions.append({
                        "type": "word_choice",
                        "message": f"Consider synonyms for '{word}' (used {count} times)",
                        "original_text": word,
                        "suggested_text": None,
                        "explanation": "Varied vocabulary makes writing more engaging.",
                        "confidence": 0.6
                    })
            
            # Check for transitional phrases
            if not self._has_adequate_transitions(doc):
                suggestions.append({
                    "type": "clarity",
                    "message": "Add transitional phrases between paragraphs",
                    "original_text": "",
                    "suggested_text": None,
                    "explanation": "Transitions help readers follow your ideas more easily.",
                    "confidence": 0.7
                })
            
            return suggestions[:10]  # Limit to 10 suggestions
            
        except Exception as e:
            logger.error("Style analysis failed", error=str(e))
            return []
    
    async def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure."""
        try:
            logger.debug("Analyzing document structure")
            
            # Detect headings
            heading_pattern = re.compile(r'^(#{1,6}\s+.+|[A-Z][A-Za-z\s]+:?)$', re.MULTILINE)
            headings = heading_pattern.findall(text)
            
            # Analyze paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            paragraph_lengths = [len(p.split()) for p in paragraphs]
            
            # Analyze sentences
            doc = self.nlp(text)
            sentences = list(doc.sents)
            sentence_lengths = [len(sent.text.split()) for sent in sentences]
            
            # Estimate reading time (average 200 words per minute)
            word_count = len([token for token in doc if not token.is_space and not token.is_punct])
            estimated_reading_time = word_count / 200
            
            return {
                "has_title": self._detect_title(text),
                "has_headings": len(headings) > 0,
                "heading_structure": [{"text": h, "level": h.count('#') if h.startswith('#') else 1} for h in headings],
                "paragraph_count": len(paragraphs),
                "paragraph_lengths": paragraph_lengths,
                "average_paragraph_length": sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0,
                "sentence_count": len(sentences),
                "sentence_lengths": sentence_lengths,
                "average_sentence_length": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
                "estimated_reading_time_minutes": round(estimated_reading_time, 1)
            }
            
        except Exception as e:
            logger.error("Structure analysis failed", error=str(e))
            return {}
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using NLTK's VADER."""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            logger.debug("Analyzing sentiment")
            
            # Initialize VADER analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Analyze overall sentiment
            overall_scores = sia.polarity_scores(text)
            
            # Analyze sentence-level sentiment
            doc = self.nlp(text)
            sentence_sentiments = []
            
            for sent in doc.sents:
                sent_scores = sia.polarity_scores(sent.text)
                sentence_sentiments.append({
                    "text": sent.text[:100],  # First 100 chars
                    "sentiment": self._interpret_sentiment(sent_scores['compound']),
                    "score": sent_scores['compound']
                })
            
            # Calculate sentiment distribution
            positive_sentences = sum(1 for s in sentence_sentiments if s['score'] > 0.1)
            negative_sentences = sum(1 for s in sentence_sentiments if s['score'] < -0.1)
            neutral_sentences = len(sentence_sentiments) - positive_sentences - negative_sentences
            
            return {
                "overall_sentiment": self._interpret_sentiment(overall_scores['compound']),
                "overall_score": round(overall_scores['compound'], 3),
                "positive_ratio": round(positive_sentences / len(sentence_sentiments), 3) if sentence_sentiments else 0,
                "negative_ratio": round(negative_sentences / len(sentence_sentiments), 3) if sentence_sentiments else 0,
                "neutral_ratio": round(neutral_sentences / len(sentence_sentiments), 3) if sentence_sentiments else 0,
                "sentence_sentiments": sentence_sentiments[:10]  # First 10 sentences
            }
            
        except Exception as e:
            logger.error("Sentiment analysis failed", error=str(e))
            return {}
    
    async def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Extract and analyze keywords."""
        try:
            logger.debug("Analyzing keywords")
            
            doc = self.nlp(text)
            
            # Extract important words (nouns, adjectives, verbs)
            important_words = [
                token.lemma_.lower() for token in doc 
                if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                not token.is_stop and 
                token.is_alpha and 
                len(token.text) > 3
            ]
            
            # Count frequency
            word_freq = Counter(important_words)
            
            # Extract named entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            entity_freq = Counter([ent[0].lower() for ent in entities])
            
            # Calculate keyword density
            total_words = len([token for token in doc if not token.is_space and not token.is_punct])
            keyword_density = len(important_words) / total_words if total_words > 0 else 0
            
            return {
                "top_keywords": word_freq.most_common(20),
                "keyword_density": round(keyword_density, 3),
                "named_entities": dict(entity_freq.most_common(10)),
                "unique_keywords": len(set(important_words)),
                "total_keywords": len(important_words)
            }
            
        except Exception as e:
            logger.error("Keyword analysis failed", error=str(e))
            return {}
    
    async def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features using spaCy."""
        try:
            logger.debug("Analyzing linguistic features")
            
            doc = self.nlp(text)
            
            # POS tag distribution
            pos_counts = Counter(token.pos_ for token in doc if not token.is_space)
            
            # Dependency analysis
            dep_counts = Counter(token.dep_ for token in doc if not token.is_space)
            
            # Calculate linguistic complexity
            subordinate_clauses = sum(1 for token in doc if token.dep_ in ['advcl', 'acl', 'relcl'])
            total_clauses = len(list(doc.sents))
            subordination_ratio = subordinate_clauses / total_clauses if total_clauses > 0 else 0
            
            return {
                "pos_distribution": dict(pos_counts.most_common()),
                "dependency_distribution": dict(dep_counts.most_common(10)),
                "subordination_ratio": round(subordination_ratio, 3),
                "average_dependency_distance": self._calculate_dependency_distance(doc),
                "linguistic_complexity_score": self._calculate_linguistic_complexity(doc)
            }
            
        except Exception as e:
            logger.error("Linguistic features analysis failed", error=str(e))
            return {}
    
    def _get_context(self, text: str, offset: int, length: int, context_size: int = 50) -> str:
        """Get context around an error."""
        start = max(0, offset - context_size)
        end = min(len(text), offset + length + context_size)
        return text[start:end]
    
    def _determine_difficulty_level(self, flesch_score: float) -> str:
        """Determine difficulty level from Flesch score."""
        for level, info in READABILITY_LEVELS.items():
            if flesch_score >= info["min_score"]:
                return level
        return "very_difficult"
    
    def _interpret_readability_score(self, flesch_score: float) -> str:
        """Interpret Flesch reading ease score."""
        level_info = READABILITY_LEVELS[self._determine_difficulty_level(flesch_score)]
        return f"Reading level: {level_info['grade']}"
    
    def _estimate_reading_time(self, text: str) -> float:
        """Estimate reading time in minutes (200 WPM average)."""
        word_count = len(text.split())
        return round(word_count / 200, 1)
    
    def _detect_passive_voice(self, doc) -> List:
        """Detect passive voice constructions."""
        passive_sentences = []
        for sent in doc.sents:
            # Look for passive voice indicators
            for token in sent:
                if token.dep_ == "auxpass" or (token.lemma_ == "be" and any(t.dep_ == "agent" for t in sent)):
                    passive_sentences.append(sent.text)
                    break
        return passive_sentences
    
    def _check_word_repetition(self, doc) -> List[Tuple[str, int]]:
        """Check for excessive word repetition."""
        words = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 4]
        word_counts = Counter(words)
        return [(word, count) for word, count in word_counts.most_common(10) if count > 3]
    
    def _has_adequate_transitions(self, doc) -> bool:
        """Check if text has adequate transitional phrases."""
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'meanwhile', 'nevertheless', 'additionally', 'similarly', 'finally'
        }
        
        text_words = set(token.lemma_.lower() for token in doc)
        transition_count = len(transition_words.intersection(text_words))
        
        paragraphs = len([sent for sent in doc.sents if sent.text.strip().endswith('\n')])
        return transition_count >= paragraphs * 0.3
    
    def _detect_title(self, text: str) -> bool:
        """Detect if text has a title."""
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a title
            return (len(first_line) < 100 and 
                   first_line.istitle() and 
                   not first_line.endswith('.'))
        return False
    
    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score."""
        if score >= 0.1:
            return "positive"
        elif score <= -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_dependency_distance(self, doc) -> float:
        """Calculate average dependency distance."""
        distances = []
        for token in doc:
            if token.head != token:  # Not root
                distance = abs(token.i - token.head.i)
                distances.append(distance)
        
        return round(sum(distances) / len(distances), 2) if distances else 0
    
    def _calculate_linguistic_complexity(self, doc) -> float:
        """Calculate overall linguistic complexity score."""
        # Combine multiple factors into a complexity score (0-100)
        factors = []
        
        # Sentence length factor
        sentences = list(doc.sents)
        if sentences:
            avg_sent_length = sum(len(sent.text.split()) for sent in sentences) / len(sentences)
            factors.append(min(avg_sent_length / 20, 1.0))  # Normalize to 0-1
        
        # Subordination factor
        subordinate_clauses = sum(1 for token in doc if token.dep_ in ['advcl', 'acl', 'relcl'])
        subordination_ratio = subordinate_clauses / len(sentences) if sentences else 0
        factors.append(min(subordination_ratio * 2, 1.0))
        
        # Vocabulary diversity factor
        words = [token.lemma_.lower() for token in doc if token.is_alpha]
        if words:
            diversity = len(set(words)) / len(words)
            factors.append(diversity)
        
        # Average and scale to 0-100
        complexity_score = sum(factors) / len(factors) if factors else 0
        return round(complexity_score * 100, 1)
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall text quality score."""
        scores = []
        
        # Readability score (inverted for difficulty)
        if results.get("readability_scores", {}).get("flesch_reading_ease"):
            readability = results["readability_scores"]["flesch_reading_ease"]
            scores.append(min(readability, 100))
        
        # Grammar score (based on issues per 100 words)
        if results.get("statistics", {}).get("word_count"):
            word_count = results["statistics"]["word_count"]
            grammar_issues = len(results.get("grammar_issues", []))
            grammar_score = max(0, 100 - (grammar_issues / word_count * 100 * 10))
            scores.append(grammar_score)
        
        # Spelling score
        if results.get("statistics", {}).get("word_count"):
            spelling_errors = len(results.get("spelling_errors", []))
            spelling_score = max(0, 100 - (spelling_errors / word_count * 100 * 5))
            scores.append(spelling_score)
        
        # Structure score (based on paragraph and sentence variety)
        if results.get("structure_analysis"):
            structure = results["structure_analysis"]
            structure_score = 70  # Base score
            if structure.get("has_headings"):
                structure_score += 10
            if structure.get("has_title"):
                structure_score += 10
            if structure.get("average_paragraph_length", 0) > 50:
                structure_score += 10
            scores.append(min(structure_score, 100))
        
        return round(sum(scores) / len(scores), 1) if scores else 50.0
    
    def _identify_improvement_areas(self, results: Dict[str, Any]) -> List[str]:
        """Identify key areas for improvement."""
        areas = []
        
        # Check readability
        readability = results.get("readability_scores", {})
        if readability.get("flesch_reading_ease", 50) < 30:
            areas.append("Improve readability - text is too complex")
        
        # Check grammar
        grammar_issues = len(results.get("grammar_issues", []))
        word_count = results.get("statistics", {}).get("word_count", 1)
        if grammar_issues / word_count > 0.02:  # More than 2% error rate
            areas.append("Fix grammar issues")
        
        # Check spelling
        spelling_errors = len(results.get("spelling_errors", []))
        if spelling_errors / word_count > 0.01:  # More than 1% error rate
            areas.append("Correct spelling errors")
        
        # Check structure
        structure = results.get("structure_analysis", {})
        if not structure.get("has_headings") and word_count > 500:
            areas.append("Add headings for better structure")
        
        # Check sentence variety
        if structure.get("average_sentence_length", 0) > 25:
            areas.append("Vary sentence length")
        
        return areas[:5]  # Top 5 areas
    
    def is_healthy(self) -> bool:
        """Check if text processor is healthy."""
        return self.is_initialized and self.nlp is not None
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.grammar_tool:
                self.grammar_tool.close()
            
            self.analysis_cache.clear()
            
            logger.info("Text processor cleanup completed")
            
        except Exception as e:
            logger.error("Text processor cleanup failed", error=str(e))
