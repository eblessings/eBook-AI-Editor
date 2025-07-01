"""
File Handler Service for the eBook Editor.
Handles file uploads, text extraction from various formats, and file processing.
Supports DOCX, PDF, EPUB, TXT, HTML, and Markdown files.
"""

import asyncio
import hashlib
import io
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO, Union
import aiofiles
import magic
import chardet
from urllib.parse import urlparse

# File processing libraries
import mammoth
import fitz  # PyMuPDF
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import docx2txt
from docx import Document
import zipfile
import structlog

from config import Settings

logger = structlog.get_logger()


class FileHandler:
    """Professional file handler with multi-format support."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.temp_files = []
        self.processing_cache = {}
        
        # Ensure directories exist
        Path(self.settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.settings.EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    
    async def extract_text_from_file(self, file) -> Dict[str, Any]:
        """Extract text content from uploaded file."""
        try:
            logger.info("Extracting text from file", 
                       filename=file.filename, 
                       content_type=file.content_type)
            
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Validate file size
            if file_size > self.settings.MAX_FILE_SIZE:
                raise ValueError(f"File too large: {file_size} bytes")
            
            # Detect actual content type
            detected_type = self._detect_content_type(content, file.filename)
            logger.debug("Content type detected", 
                        original=file.content_type, 
                        detected=detected_type)
            
            # Generate file ID and save temporarily
            file_id = str(uuid.uuid4())
            temp_path = await self._save_temp_file(content, file_id, file.filename)
            
            # Extract text based on file type
            extracted_data = await self._extract_by_type(
                content, temp_path, detected_type, file.filename
            )
            
            # Calculate additional metadata
            word_count = len(extracted_data['text'].split())
            char_count = len(extracted_data['text'])
            
            # Clean up temp file
            await self._cleanup_temp_file(temp_path)
            
            result = {
                'file_id': file_id,
                'filename': file.filename,
                'original_type': file.content_type,
                'detected_type': detected_type,
                'file_size': file_size,
                'text': extracted_data['text'],
                'word_count': word_count,
                'character_count': char_count,
                'metadata': extracted_data.get('metadata', {}),
                'structure': extracted_data.get('structure', {}),
                'images': extracted_data.get('images', []),
                'extraction_method': extracted_data.get('method', 'unknown')
            }
            
            logger.info("Text extraction completed", 
                       file_id=file_id, 
                       word_count=word_count)
            
            return result
            
        except Exception as e:
            logger.error("Text extraction failed", 
                        filename=file.filename, 
                        error=str(e))
            raise
    
    def _detect_content_type(self, content: bytes, filename: str) -> str:
        """Detect actual content type using multiple methods."""
        try:
            # Try python-magic first
            detected_mime = magic.from_buffer(content, mime=True)
            if detected_mime and detected_mime != 'application/octet-stream':
                return detected_mime
        except:
            pass
        
        # Fallback to filename extension
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
        
        # Check file signatures
        if content.startswith(b'PK\x03\x04'):
            if b'word/' in content[:1024]:
                return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif b'epub' in content[:1024].lower():
                return 'application/epub+zip'
        elif content.startswith(b'%PDF'):
            return 'application/pdf'
        elif content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff'):
            return 'text/plain'  # UTF-16 text
        
        # Default fallback
        return 'application/octet-stream'
    
    async def _save_temp_file(self, content: bytes, file_id: str, filename: str) -> str:
        """Save content to temporary file."""
        temp_path = os.path.join(
            self.settings.TEMP_DIR, 
            f"{file_id}_{filename}"
        )
        
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        self.temp_files.append(temp_path)
        return temp_path
    
    async def _cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file."""
        try:
            if temp_path in self.temp_files:
                self.temp_files.remove(temp_path)
            await aiofiles.os.remove(temp_path)
        except Exception as e:
            logger.warning("Failed to cleanup temp file", path=temp_path, error=str(e))
    
    async def _extract_by_type(self, content: bytes, temp_path: str, 
                              content_type: str, filename: str) -> Dict[str, Any]:
        """Extract text based on content type."""
        try:
            # Route to appropriate extraction method
            if content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                return await self._extract_from_docx(temp_path)
            elif content_type == 'application/pdf':
                return await self._extract_from_pdf(temp_path)
            elif content_type == 'application/epub+zip':
                return await self._extract_from_epub(temp_path)
            elif content_type in ['text/plain', 'text/markdown']:
                return await self._extract_from_text(content)
            elif content_type == 'text/html':
                return await self._extract_from_html(content)
            else:
                # Try as text
                logger.warning("Unknown content type, trying as text", 
                              content_type=content_type)
                return await self._extract_from_text(content)
                
        except Exception as e:
            logger.error("Type-specific extraction failed", 
                        content_type=content_type, 
                        error=str(e))
            # Fallback to basic text extraction
            return await self._extract_from_text(content)
    
    async def _extract_from_docx(self, file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX file."""
        try:
            logger.debug("Extracting from DOCX")
            
            # Use mammoth for better HTML conversion
            with open(file_path, 'rb') as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html_content = result.value
                
                # Convert HTML to plain text while preserving structure
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text('\n\n', strip=True)
            
            # Also extract metadata using python-docx
            try:
                doc = Document(file_path)
                metadata = {
                    'title': doc.core_properties.title or '',
                    'author': doc.core_properties.author or '',
                    'subject': doc.core_properties.subject or '',
                    'created': str(doc.core_properties.created) if doc.core_properties.created else '',
                    'modified': str(doc.core_properties.modified) if doc.core_properties.modified else ''
                }
                
                # Extract structure information
                structure = {
                    'paragraphs': len(doc.paragraphs),
                    'tables': len(doc.tables),
                    'sections': len(doc.sections)
                }
                
            except Exception as e:
                logger.warning("Failed to extract DOCX metadata", error=str(e))
                metadata = {}
                structure = {}
            
            return {
                'text': text_content,
                'metadata': metadata,
                'structure': structure,
                'method': 'mammoth+python-docx',
                'html_content': html_content
            }
            
        except Exception as e:
            logger.error("DOCX extraction failed", error=str(e))
            # Fallback to simpler extraction
            try:
                text = docx2txt.process(file_path)
                return {
                    'text': text,
                    'metadata': {},
                    'structure': {},
                    'method': 'docx2txt'
                }
            except:
                raise ValueError(f"Failed to extract text from DOCX: {str(e)}")
    
    async def _extract_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file."""
        try:
            logger.debug("Extracting from PDF")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            
            text_content = []
            metadata = doc.metadata
            images = []
            structure = {
                'pages': doc.page_count,
                'has_toc': len(doc.get_toc()) > 0,
                'toc_entries': len(doc.get_toc())
            }
            
            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():
                    text_content.append(page_text)
                
                # Extract images (metadata only)
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    images.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'xref': img[0]
                    })
            
            doc.close()
            
            # Join all text with page breaks
            full_text = '\n\n--- Page Break ---\n\n'.join(text_content)
            
            return {
                'text': full_text,
                'metadata': {
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'subject': metadata.get('subject', ''),
                    'creator': metadata.get('creator', ''),
                    'producer': metadata.get('producer', ''),
                    'creation_date': metadata.get('creationDate', ''),
                    'modification_date': metadata.get('modDate', '')
                },
                'structure': structure,
                'images': images,
                'method': 'pymupdf'
            }
            
        except Exception as e:
            logger.error("PDF extraction failed", error=str(e))
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    async def _extract_from_epub(self, file_path: str) -> Dict[str, Any]:
        """Extract text from EPUB file."""
        try:
            logger.debug("Extracting from EPUB")
            
            # Open EPUB with ebooklib
            book = epub.read_epub(file_path)
            
            # Extract metadata
            metadata = {
                'title': book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else '',
                'author': book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else '',
                'language': book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else '',
                'publisher': book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else '',
                'description': book.get_metadata('DC', 'description')[0][0] if book.get_metadata('DC', 'description') else '',
                'identifier': book.get_metadata('DC', 'identifier')[0][0] if book.get_metadata('DC', 'identifier') else ''
            }
            
            # Extract text from all document items
            text_content = []
            images = []
            chapter_count = 0
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    chapter_text = soup.get_text('\n', strip=True)
                    
                    if chapter_text.strip():
                        text_content.append(chapter_text)
                        chapter_count += 1
                        
                elif item.get_type() == ebooklib.ITEM_IMAGE:
                    images.append({
                        'filename': item.get_name(),
                        'media_type': item.get_type()
                    })
            
            # Join chapters
            full_text = '\n\n--- Chapter Break ---\n\n'.join(text_content)
            
            structure = {
                'chapters': chapter_count,
                'images': len(images),
                'has_toc': len(book.toc) > 0
            }
            
            return {
                'text': full_text,
                'metadata': metadata,
                'structure': structure,
                'images': images,
                'method': 'ebooklib'
            }
            
        except Exception as e:
            logger.error("EPUB extraction failed", error=str(e))
            raise ValueError(f"Failed to extract text from EPUB: {str(e)}")
    
    async def _extract_from_text(self, content: bytes) -> Dict[str, Any]:
        """Extract text from plain text file."""
        try:
            logger.debug("Extracting from text file")
            
            # Detect encoding
            detected = chardet.detect(content)
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 0)
            
            logger.debug("Encoding detected", encoding=encoding, confidence=confidence)
            
            # Decode text
            try:
                text = content.decode(encoding)
            except UnicodeDecodeError:
                # Fallback encodings
                for fallback_encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        text = content.decode(fallback_encoding)
                        encoding = fallback_encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # Last resort: decode with errors ignored
                    text = content.decode('utf-8', errors='ignore')
                    encoding = 'utf-8 (with errors ignored)'
            
            # Basic structure analysis
            lines = text.split('\n')
            paragraphs = text.split('\n\n')
            
            structure = {
                'lines': len(lines),
                'paragraphs': len([p for p in paragraphs if p.strip()]),
                'encoding': encoding,
                'encoding_confidence': confidence
            }
            
            return {
                'text': text,
                'metadata': {'encoding': encoding},
                'structure': structure,
                'method': 'chardet+decode'
            }
            
        except Exception as e:
            logger.error("Text extraction failed", error=str(e))
            raise ValueError(f"Failed to extract text: {str(e)}")
    
    async def _extract_from_html(self, content: bytes) -> Dict[str, Any]:
        """Extract text from HTML file."""
        try:
            logger.debug("Extracting from HTML")
            
            # Detect encoding
            detected = chardet.detect(content)
            encoding = detected.get('encoding', 'utf-8')
            
            # Decode HTML
            html_text = content.decode(encoding)
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # Extract metadata from meta tags
            metadata = {}
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Look for common meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                content_attr = meta.get('content', '')
                
                if name in ['author', 'description', 'keywords']:
                    metadata[name] = content_attr
            
            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text('\n', strip=True)
            
            # Structure analysis
            structure = {
                'headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                'paragraphs': len(soup.find_all('p')),
                'links': len(soup.find_all('a')),
                'images': len(soup.find_all('img')),
                'encoding': encoding
            }
            
            return {
                'text': text,
                'metadata': metadata,
                'structure': structure,
                'method': 'beautifulsoup',
                'html_content': html_text
            }
            
        except Exception as e:
            logger.error("HTML extraction failed", error=str(e))
            raise ValueError(f"Failed to extract text from HTML: {str(e)}")
    
    async def start_batch_processing(self, file) -> str:
        """Start batch processing for a file."""
        try:
            task_id = str(uuid.uuid4())
            
            # In a real implementation, this would queue the file for background processing
            logger.info("Starting batch processing", task_id=task_id, filename=file.filename)
            
            # For now, just return the task ID
            return task_id
            
        except Exception as e:
            logger.error("Failed to start batch processing", error=str(e))
            raise
    
    async def validate_file(self, file, max_size: Optional[int] = None) -> Dict[str, Any]:
        """Validate uploaded file."""
        try:
            # Check file size
            content = await file.read()
            file_size = len(content)
            
            max_allowed = max_size or self.settings.MAX_FILE_SIZE
            if file_size > max_allowed:
                return {
                    'valid': False,
                    'error': f'File too large: {file_size} bytes (max: {max_allowed})'
                }
            
            # Check content type
            detected_type = self._detect_content_type(content, file.filename)
            if detected_type not in self.settings.SUPPORTED_UPLOAD_TYPES:
                return {
                    'valid': False,
                    'error': f'Unsupported file type: {detected_type}'
                }
            
            # Check for malicious content (basic checks)
            if self._is_potentially_malicious(content, file.filename):
                return {
                    'valid': False,
                    'error': 'File failed security checks'
                }
            
            return {
                'valid': True,
                'file_size': file_size,
                'detected_type': detected_type,
                'original_type': file.content_type
            }
            
        except Exception as e:
            logger.error("File validation failed", error=str(e))
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def _is_potentially_malicious(self, content: bytes, filename: str) -> bool:
        """Basic malicious content detection."""
        # Check for suspicious file extensions
        suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.jar']
        if any(filename.lower().endswith(ext) for ext in suspicious_extensions):
            return True
        
        # Check for executable signatures
        if content.startswith(b'MZ'):  # PE executable
            return True
        
        # Check for script content in non-script files
        if b'<script' in content.lower() and not filename.lower().endswith('.html'):
            return True
        
        return False
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Basic file info
            stat = path.stat()
            
            # Read content for type detection
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read(1024)  # First 1KB for type detection
            
            detected_type = self._detect_content_type(content, path.name)
            
            return {
                'path': str(path.absolute()),
                'name': path.name,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'detected_type': detected_type,
                'extension': path.suffix.lower(),
                'is_supported': detected_type in self.settings.SUPPORTED_UPLOAD_TYPES
            }
            
        except Exception as e:
            logger.error("Failed to get file info", file_path=file_path, error=str(e))
            raise
    
    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old temporary files."""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            temp_dir = Path(self.settings.TEMP_DIR)
            deleted_count = 0
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning("Failed to delete old file", 
                                     file=str(file_path), error=str(e))
            
            logger.info("Cleanup completed", deleted_files=deleted_count)
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
    
    def cleanup(self):
        """Clean up resources and temporary files."""
        try:
            for temp_file in self.temp_files.copy():
                try:
                    Path(temp_file).unlink(missing_ok=True)
                    self.temp_files.remove(temp_file)
                except Exception as e:
                    logger.warning("Failed to cleanup temp file", 
                                 file=temp_file, error=str(e))
            
            self.processing_cache.clear()
            logger.info("File handler cleanup completed")
            
        except Exception as e:
            logger.error("File handler cleanup failed", error=str(e))
