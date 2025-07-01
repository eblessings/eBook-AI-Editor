"""
Professional eBook Generation Service.
Creates EPUB, PDF, DOCX, and other formats using ebooklib, reportlab, and other libraries.
Supports AI-enhanced content processing and professional formatting.
"""

import asyncio
import io
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO, Union
import base64

from ebooklib import epub
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from jinja2 import Template
from PIL import Image
import structlog

from config import Settings, EBOOK_FORMATS
from api.models import EBookMetadata, FormatOptions, ChapterConfiguration, AIEnhancementOptions

logger = structlog.get_logger()


class EBookGenerator:
    """Professional eBook generator with multiple format support."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.temp_files = []
        
        # HTML templates for eBook content
        self.chapter_template = Template("""
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml">
        <head>
            <title>{{ title }}</title>
            <meta charset="utf-8"/>
            <link rel="stylesheet" type="text/css" href="style/main.css"/>
        </head>
        <body>
            <div class="chapter">
                <h1 class="chapter-title">{{ title }}</h1>
                <div class="chapter-content">
                    {{ content | safe }}
                </div>
            </div>
        </body>
        </html>
        """)
        
        self.css_template = Template("""
        body {
            font-family: {{ font_family }};
            font-size: {{ font_size }}pt;
            line-height: {{ line_height }};
            margin: {{ margin_top }}px {{ margin_right }}px {{ margin_bottom }}px {{ margin_left }}px;
            text-align: {{ 'justify' if justify_text else 'left' }};
        }
        
        .chapter {
            page-break-before: {{ 'always' if page_break_before_chapter else 'auto' }};
        }
        
        .chapter-title {
            font-size: 1.5em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2em;
            page-break-after: avoid;
        }
        
        .chapter-content {
            text-indent: 1.5em;
        }
        
        .chapter-content p {
            margin-bottom: 1em;
        }
        
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        
        blockquote {
            margin: 1em 2em;
            font-style: italic;
            border-left: 3px solid #ccc;
            padding-left: 1em;
        }
        
        .toc {
            page-break-after: always;
        }
        
        .toc-entry {
            margin-bottom: 0.5em;
        }
        
        .front-matter, .back-matter {
            page-break-before: always;
        }
        """)
    
    async def create_ebook(self, content: str, metadata: EBookMetadata, 
                          format_options: FormatOptions, ai_options: AIEnhancementOptions) -> BinaryIO:
        """Create eBook in the specified format."""
        try:
            logger.info("Starting eBook generation", 
                       title=metadata.title, 
                       format=format_options.__dict__.get('format', 'epub'))
            
            # Process and enhance content
            processed_content = await self._preprocess_content(content, ai_options)
            
            # Detect chapters
            chapters = await self._detect_chapters(processed_content, metadata)
            
            # Generate eBook based on format (defaulting to EPUB)
            format_type = getattr(format_options, 'format', 'epub')
            
            if format_type == 'epub':
                return await self._generate_epub(chapters, metadata, format_options)
            elif format_type == 'pdf':
                return await self._generate_pdf(chapters, metadata, format_options)
            elif format_type == 'docx':
                return await self._generate_docx(chapters, metadata, format_options)
            elif format_type == 'html':
                return await self._generate_html(chapters, metadata, format_options)
            elif format_type == 'txt':
                return await self._generate_txt(chapters, metadata, format_options)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error("eBook generation failed", error=str(e))
            raise
    
    async def _preprocess_content(self, content: str, ai_options: AIEnhancementOptions) -> str:
        """Preprocess and enhance content before eBook generation."""
        try:
            processed_content = content
            
            if ai_options.enhance_before_generation:
                logger.info("Applying AI enhancements to content")
                
                # Apply various enhancements based on options
                if ai_options.auto_correct_spelling:
                    processed_content = await self._auto_correct_spelling(processed_content)
                
                if ai_options.improve_grammar:
                    processed_content = await self._improve_grammar(processed_content)
                
                if ai_options.enhance_style:
                    processed_content = await self._enhance_style(processed_content, ai_options.enhancement_strength)
                
                if ai_options.improve_readability:
                    processed_content = await self._improve_readability(processed_content)
            
            # Clean and format content
            processed_content = self._clean_content(processed_content)
            
            return processed_content
            
        except Exception as e:
            logger.error("Content preprocessing failed", error=str(e))
            return content  # Return original if preprocessing fails
    
    async def _detect_chapters(self, content: str, metadata: EBookMetadata) -> List[Dict[str, Any]]:
        """Detect and structure chapters from content."""
        try:
            logger.info("Detecting chapters in content")
            
            # For now, implement a simple chapter detection
            # In a real implementation, this would use AI or more sophisticated methods
            
            # Split by common chapter indicators
            chapter_patterns = [
                r'\n\s*Chapter\s+\d+[:\s]',
                r'\n\s*CHAPTER\s+\d+[:\s]',
                r'\n\s*\d+\.\s+',
                r'\n\s*[A-Z][A-Z\s]{10,}\n',  # All caps headings
            ]
            
            chapters = []
            
            # Try to split by patterns
            import re
            for pattern in chapter_patterns:
                matches = list(re.finditer(pattern, content))
                if len(matches) > 1:  # Found multiple chapters
                    for i, match in enumerate(matches):
                        start = match.start()
                        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                        
                        chapter_content = content[start:end].strip()
                        chapter_title = self._extract_chapter_title(chapter_content)
                        
                        chapters.append({
                            "title": chapter_title or f"Chapter {i + 1}",
                            "content": self._clean_chapter_content(chapter_content),
                            "word_count": len(chapter_content.split()),
                            "order": i + 1
                        })
                    break
            
            # If no chapters detected, create a single chapter
            if not chapters:
                chapters = [{
                    "title": metadata.title or "Main Content",
                    "content": self._clean_chapter_content(content),
                    "word_count": len(content.split()),
                    "order": 1
                }]
            
            logger.info("Chapter detection completed", chapter_count=len(chapters))
            return chapters
            
        except Exception as e:
            logger.error("Chapter detection failed", error=str(e))
            # Fallback: single chapter
            return [{
                "title": metadata.title or "Main Content",
                "content": content,
                "word_count": len(content.split()),
                "order": 1
            }]
    
    async def _generate_epub(self, chapters: List[Dict], metadata: EBookMetadata, 
                           format_options: FormatOptions) -> BinaryIO:
        """Generate EPUB format eBook."""
        try:
            logger.info("Generating EPUB", chapter_count=len(chapters))
            
            # Create EPUB book
            book = epub.EpubBook()
            
            # Set metadata
            book.set_identifier(metadata.isbn or str(uuid.uuid4()))
            book.set_title(metadata.title)
            book.set_language(metadata.language)
            book.add_author(metadata.author)
            
            if metadata.description:
                book.add_metadata('DC', 'description', metadata.description)
            if metadata.publisher:
                book.add_metadata('DC', 'publisher', metadata.publisher)
            if metadata.genre:
                book.add_metadata('DC', 'subject', metadata.genre)
            
            book.add_metadata('DC', 'date', datetime.now().isoformat())
            
            # Create CSS file
            css_content = self.css_template.render(
                font_family=format_options.font_family,
                font_size=format_options.font_size,
                line_height=format_options.line_height,
                margin_top=format_options.margin_top,
                margin_right=format_options.margin_right,
                margin_bottom=format_options.margin_bottom,
                margin_left=format_options.margin_left,
                justify_text=format_options.justify_text,
                page_break_before_chapter=format_options.page_break_before_chapter
            )
            
            nav_css = epub.EpubItem(
                uid="style_nav",
                file_name="style/main.css",
                media_type="text/css",
                content=css_content
            )
            book.add_item(nav_css)
            
            # Add cover if specified
            if metadata.cover_image_url and format_options.include_cover:
                await self._add_cover_to_epub(book, metadata.cover_image_url)
            
            # Create chapters
            epub_chapters = []
            spine_items = ['nav']
            
            for i, chapter in enumerate(chapters):
                chapter_html = self.chapter_template.render(
                    title=chapter['title'],
                    content=self._format_content_as_html(chapter['content'])
                )
                
                epub_chapter = epub.EpubHtml(
                    title=chapter['title'],
                    file_name=f'chap_{i+1:02d}.xhtml',
                    lang=metadata.language
                )
                epub_chapter.content = chapter_html
                epub_chapter.add_item(nav_css)
                
                book.add_item(epub_chapter)
                epub_chapters.append(epub_chapter)
                spine_items.append(epub_chapter)
            
            # Define Table of Contents
            if format_options.include_toc:
                book.toc = tuple(
                    epub.Link(f'chap_{i+1:02d}.xhtml', chapter['title'], f'chap_{i+1}')
                    for i, chapter in enumerate(chapters)
                )
            
            # Add navigation files
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            
            # Define spine
            book.spine = spine_items
            
            # Generate EPUB file
            epub_content = io.BytesIO()
            epub.write_epub(epub_content, book, {})
            epub_content.seek(0)
            
            logger.info("EPUB generation completed")
            return epub_content
            
        except Exception as e:
            logger.error("EPUB generation failed", error=str(e))
            raise
    
    async def _generate_pdf(self, chapters: List[Dict], metadata: EBookMetadata, 
                          format_options: FormatOptions) -> BinaryIO:
        """Generate PDF format eBook using PyMuPDF."""
        try:
            logger.info("Generating PDF", chapter_count=len(chapters))
            
            # Create new PDF document
            doc = fitz.open()
            
            # Set metadata
            doc.set_metadata({
                "title": metadata.title,
                "author": metadata.author,
                "subject": metadata.description or "",
                "creator": "eBook Editor Pro",
                "producer": metadata.publisher or "eBook Editor Pro"
            })
            
            # Font and formatting setup
            font_size = format_options.font_size
            line_height = font_size * format_options.line_height
            
            # Add cover page if requested
            if format_options.include_cover:
                self._add_pdf_cover_page(doc, metadata, format_options)
            
            # Add table of contents if requested
            if format_options.include_toc and len(chapters) > 1:
                self._add_pdf_toc(doc, chapters, format_options)
            
            # Add chapters
            for chapter in chapters:
                if format_options.page_break_before_chapter and doc.page_count > 0:
                    doc.new_page()
                
                self._add_pdf_chapter(doc, chapter, format_options)
            
            # Save PDF to memory
            pdf_content = io.BytesIO()
            pdf_bytes = doc.tobytes()
            pdf_content.write(pdf_bytes)
            pdf_content.seek(0)
            
            doc.close()
            
            logger.info("PDF generation completed")
            return pdf_content
            
        except Exception as e:
            logger.error("PDF generation failed", error=str(e))
            raise
    
    async def _generate_docx(self, chapters: List[Dict], metadata: EBookMetadata, 
                           format_options: FormatOptions) -> BinaryIO:
        """Generate DOCX format eBook."""
        try:
            logger.info("Generating DOCX", chapter_count=len(chapters))
            
            # Create new document
            doc = Document()
            
            # Set document properties
            properties = doc.core_properties
            properties.title = metadata.title
            properties.author = metadata.author
            if metadata.description:
                properties.comments = metadata.description
            if metadata.publisher:
                properties.category = metadata.publisher
            
            # Configure styles
            style = doc.styles['Normal']
            font = style.font
            font.name = format_options.font_family
            font.size = Pt(format_options.font_size)
            
            paragraph_format = style.paragraph_format
            paragraph_format.line_spacing = format_options.line_height
            paragraph_format.space_after = Pt(6)
            
            # Add title page if cover is requested
            if format_options.include_cover:
                self._add_docx_title_page(doc, metadata)
                doc.add_page_break()
            
            # Add table of contents if requested
            if format_options.include_toc and len(chapters) > 1:
                self._add_docx_toc(doc, chapters)
                doc.add_page_break()
            
            # Add chapters
            for i, chapter in enumerate(chapters):
                if i > 0 and format_options.page_break_before_chapter:
                    doc.add_page_break()
                
                # Chapter title
                title_paragraph = doc.add_heading(chapter['title'], level=1)
                title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # Chapter content
                content_paragraphs = chapter['content'].split('\n\n')
                for para_text in content_paragraphs:
                    if para_text.strip():
                        paragraph = doc.add_paragraph(para_text.strip())
                        if format_options.justify_text:
                            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            
            # Save to memory
            docx_content = io.BytesIO()
            doc.save(docx_content)
            docx_content.seek(0)
            
            logger.info("DOCX generation completed")
            return docx_content
            
        except Exception as e:
            logger.error("DOCX generation failed", error=str(e))
            raise
    
    async def _generate_html(self, chapters: List[Dict], metadata: EBookMetadata, 
                           format_options: FormatOptions) -> BinaryIO:
        """Generate HTML format eBook."""
        try:
            logger.info("Generating HTML", chapter_count=len(chapters))
            
            # HTML template
            html_template = Template("""
            <!DOCTYPE html>
            <html lang="{{ language }}">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ title }}</title>
                <meta name="author" content="{{ author }}">
                {% if description %}<meta name="description" content="{{ description }}">{% endif %}
                <style>{{ css }}</style>
            </head>
            <body>
                <header class="book-header">
                    <h1>{{ title }}</h1>
                    <p class="author">by {{ author }}</p>
                    {% if description %}<p class="description">{{ description }}</p>{% endif %}
                </header>
                
                {% if include_toc and chapters|length > 1 %}
                <nav class="toc">
                    <h2>Table of Contents</h2>
                    <ul>
                    {% for chapter in chapters %}
                        <li><a href="#chapter-{{ loop.index }}">{{ chapter.title }}</a></li>
                    {% endfor %}
                    </ul>
                </nav>
                {% endif %}
                
                <main class="book-content">
                {% for chapter in chapters %}
                    <section class="chapter" id="chapter-{{ loop.index }}">
                        <h2 class="chapter-title">{{ chapter.title }}</h2>
                        <div class="chapter-content">
                            {{ chapter.formatted_content | safe }}
                        </div>
                    </section>
                {% endfor %}
                </main>
                
                <footer class="book-footer">
                    <p>Generated by eBook Editor Pro</p>
                </footer>
            </body>
            </html>
            """)
            
            # Generate CSS
            css_content = self.css_template.render(
                font_family=format_options.font_family,
                font_size=format_options.font_size,
                line_height=format_options.line_height,
                margin_top=format_options.margin_top,
                margin_right=format_options.margin_right,
                margin_bottom=format_options.margin_bottom,
                margin_left=format_options.margin_left,
                justify_text=format_options.justify_text,
                page_break_before_chapter=format_options.page_break_before_chapter
            ) + """
            .book-header { text-align: center; margin-bottom: 3em; }
            .author { font-style: italic; font-size: 1.2em; }
            .description { margin-top: 1em; font-style: italic; }
            .toc { margin-bottom: 3em; }
            .toc ul { list-style-type: none; }
            .toc li { margin-bottom: 0.5em; }
            .chapter { margin-bottom: 3em; }
            .book-footer { margin-top: 3em; text-align: center; font-size: 0.9em; color: #666; }
            """
            
            # Format chapters
            formatted_chapters = []
            for chapter in chapters:
                formatted_chapters.append({
                    'title': chapter['title'],
                    'formatted_content': self._format_content_as_html(chapter['content'])
                })
            
            # Render HTML
            html_content = html_template.render(
                title=metadata.title,
                author=metadata.author,
                description=metadata.description,
                language=metadata.language,
                css=css_content,
                chapters=formatted_chapters,
                include_toc=format_options.include_toc
            )
            
            # Return as BytesIO
            html_bytes = io.BytesIO()
            html_bytes.write(html_content.encode('utf-8'))
            html_bytes.seek(0)
            
            logger.info("HTML generation completed")
            return html_bytes
            
        except Exception as e:
            logger.error("HTML generation failed", error=str(e))
            raise
    
    async def _generate_txt(self, chapters: List[Dict], metadata: EBookMetadata, 
                          format_options: FormatOptions) -> BinaryIO:
        """Generate plain text format eBook."""
        try:
            logger.info("Generating TXT", chapter_count=len(chapters))
            
            content_lines = []
            
            # Add header
            content_lines.extend([
                metadata.title,
                f"by {metadata.author}",
                ""
            ])
            
            if metadata.description:
                content_lines.extend([metadata.description, ""])
            
            # Add table of contents if requested
            if format_options.include_toc and len(chapters) > 1:
                content_lines.extend(["TABLE OF CONTENTS", ""])
                for i, chapter in enumerate(chapters, 1):
                    content_lines.append(f"{i}. {chapter['title']}")
                content_lines.extend(["", "=" * 50, ""])
            
            # Add chapters
            for i, chapter in enumerate(chapters):
                if i > 0:
                    content_lines.extend(["", "=" * 50, ""])
                
                content_lines.extend([
                    chapter['title'],
                    "-" * len(chapter['title']),
                    ""
                ])
                
                # Format content
                paragraphs = chapter['content'].split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        # Wrap text to reasonable line length
                        lines = self._wrap_text(paragraph.strip(), 80)
                        content_lines.extend(lines)
                        content_lines.append("")
            
            # Join all content
            full_content = '\n'.join(content_lines)
            
            # Return as BytesIO
            txt_bytes = io.BytesIO()
            txt_bytes.write(full_content.encode('utf-8'))
            txt_bytes.seek(0)
            
            logger.info("TXT generation completed")
            return txt_bytes
            
        except Exception as e:
            logger.error("TXT generation failed", error=str(e))
            raise
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Normalize quotes
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        
        # Remove excessive spaces
        content = re.sub(r' +', ' ', content)
        
        return content.strip()
    
    def _extract_chapter_title(self, chapter_content: str) -> Optional[str]:
        """Extract chapter title from content."""
        lines = chapter_content.split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and not first_line.endswith('.'):
                return first_line
        return None
    
    def _clean_chapter_content(self, content: str) -> str:
        """Clean chapter content, removing title if present."""
        lines = content.split('\n')
        # Remove first line if it looks like a title
        if lines and len(lines[0].strip()) < 100 and not lines[0].strip().endswith('.'):
            lines = lines[1:]
        
        return '\n'.join(lines).strip()
    
    def _format_content_as_html(self, content: str) -> str:
        """Format plain text content as HTML."""
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        html_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if para:
                # Simple formatting
                para = para.replace('\n', '<br>')
                
                # Basic markdown-like formatting
                para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para)
                para = re.sub(r'\*(.*?)\*', r'<em>\1</em>', para)
                
                html_paragraphs.append(f'<p>{para}</p>')
        
        return '\n'.join(html_paragraphs)
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        import textwrap
        return textwrap.wrap(text, width=width)
    
    async def _add_cover_to_epub(self, book: epub.EpubBook, cover_url: str):
        """Add cover image to EPUB."""
        try:
            # This would download and add cover image
            # For now, just add a placeholder
            pass
        except Exception as e:
            logger.warning("Failed to add cover to EPUB", error=str(e))
    
    def _add_pdf_cover_page(self, doc, metadata: EBookMetadata, format_options: FormatOptions):
        """Add cover page to PDF."""
        page = doc.new_page()
        
        # Add title
        rect = fitz.Rect(50, 200, 500, 300)
        page.insert_textbox(rect, metadata.title, 
                           fontsize=24, fontname="helv", 
                           align=fitz.TEXT_ALIGN_CENTER)
        
        # Add author
        rect = fitz.Rect(50, 320, 500, 360)
        page.insert_textbox(rect, f"by {metadata.author}", 
                           fontsize=16, fontname="helv", 
                           align=fitz.TEXT_ALIGN_CENTER)
    
    def _add_pdf_toc(self, doc, chapters: List[Dict], format_options: FormatOptions):
        """Add table of contents to PDF."""
        page = doc.new_page()
        
        # Title
        rect = fitz.Rect(50, 50, 500, 100)
        page.insert_textbox(rect, "Table of Contents", 
                           fontsize=18, fontname="helv", 
                           align=fitz.TEXT_ALIGN_CENTER)
        
        # Chapters
        y_pos = 120
        for i, chapter in enumerate(chapters, 1):
            rect = fitz.Rect(50, y_pos, 500, y_pos + 20)
            page.insert_textbox(rect, f"{i}. {chapter['title']}", 
                               fontsize=12, fontname="helv")
            y_pos += 25
    
    def _add_pdf_chapter(self, doc, chapter: Dict, format_options: FormatOptions):
        """Add chapter content to PDF."""
        page = doc.new_page()
        
        # Chapter title
        rect = fitz.Rect(50, 50, 500, 100)
        page.insert_textbox(rect, chapter['title'], 
                           fontsize=16, fontname="helvb", 
                           align=fitz.TEXT_ALIGN_CENTER)
        
        # Chapter content
        content_rect = fitz.Rect(50, 120, 500, 750)
        page.insert_textbox(content_rect, chapter['content'], 
                           fontsize=format_options.font_size, 
                           fontname="helv")
    
    def _add_docx_title_page(self, doc: Document, metadata: EBookMetadata):
        """Add title page to DOCX."""
        # Title
        title = doc.add_heading(metadata.title, 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add some space
        doc.add_paragraph()
        doc.add_paragraph()
        
        # Author
        author = doc.add_paragraph(f"by {metadata.author}")
        author.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Description if available
        if metadata.description:
            doc.add_paragraph()
            desc = doc.add_paragraph(metadata.description)
            desc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    def _add_docx_toc(self, doc: Document, chapters: List[Dict]):
        """Add table of contents to DOCX."""
        doc.add_heading("Table of Contents", level=1)
        
        for i, chapter in enumerate(chapters, 1):
            toc_entry = doc.add_paragraph(f"{i}. {chapter['title']}")
            toc_entry.style = 'List Number'
    
    # Placeholder methods for AI enhancement (would integrate with AI service)
    async def _auto_correct_spelling(self, content: str) -> str:
        """Auto-correct spelling errors."""
        # Placeholder - would integrate with spell checker
        return content
    
    async def _improve_grammar(self, content: str) -> str:
        """Improve grammar."""
        # Placeholder - would integrate with grammar checker
        return content
    
    async def _enhance_style(self, content: str, strength: str) -> str:
        """Enhance writing style."""
        # Placeholder - would integrate with AI service
        return content
    
    async def _improve_readability(self, content: str) -> str:
        """Improve readability."""
        # Placeholder - would integrate with AI service
        return content
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete temp file", file=temp_file, error=str(e))
        self.temp_files.clear()
