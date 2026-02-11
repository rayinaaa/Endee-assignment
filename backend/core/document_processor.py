"""
Document Processing Pipeline
Extracts and processes scientific papers (PDFs) for vector indexing
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

import PyPDF2
import fitz  # PyMuPDF for better PDF processing
from sentence_transformers import util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    chunk_id: str
    text: str
    chunk_type: str  # paragraph, section, figure_caption, etc.
    page_number: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

@dataclass  
class ProcessedDocument:
    """Represents a fully processed document"""
    doc_id: str
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    chunks: List[DocumentChunk]
    full_text: str
    metadata: Dict[str, Any]
    
    def get_full_text(self) -> str:
        """Get the full document text"""
        return self.full_text


class DocumentProcessor:
    """Processes scientific documents for RAG indexing"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Load spaCy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic text processing.")
            self.nlp = None
        
        # Initialize text vectorizer for chunk boundary detection
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        logger.info(f"DocumentProcessor initialized (chunk_size: {chunk_size}, overlap: {overlap})")
    
    async def process_pdf(self, file_path: str, metadata: Optional[str] = None) -> ProcessedDocument:
        """Process a PDF document"""
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Extract text and metadata from PDF
            pdf_content = await self._extract_pdf_content(file_path)
            
            # Parse additional metadata if provided
            extra_metadata = json.loads(metadata) if metadata else {}
            
            # Extract document metadata
            doc_metadata = await self._extract_document_metadata(
                pdf_content["text"], 
                pdf_content["metadata"],
                extra_metadata
            )
            
            # Generate document ID
            doc_id = self._generate_doc_id(doc_metadata["title"])
            
            # Process and chunk the text
            chunks = await self._chunk_document(
                pdf_content["text"], 
                pdf_content["pages"],
                doc_id
            )
            
            # Create processed document
            processed_doc = ProcessedDocument(
                doc_id=doc_id,
                title=doc_metadata["title"],
                authors=doc_metadata["authors"],
                abstract=doc_metadata["abstract"],
                year=doc_metadata.get("year"),
                venue=doc_metadata.get("venue"),
                doi=doc_metadata.get("doi"),
                chunks=chunks,
                full_text=pdf_content["text"],
                metadata={
                    **doc_metadata,
                    **extra_metadata,
                    "processed_at": datetime.utcnow().isoformat(),
                    "chunk_count": len(chunks),
                    "file_path": file_path
                }
            )
            
            logger.info(f"Processed document: {doc_id} ({len(chunks)} chunks)")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    async def _extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from PDF"""
        text_content = ""
        pages = []
        metadata = {}
        
        try:
            # Use PyMuPDF for better text extraction
            doc = fitz.open(file_path)
            
            # Extract PDF metadata
            pdf_metadata = doc.metadata
            metadata.update({
                "pdf_title": pdf_metadata.get("title", ""),
                "pdf_author": pdf_metadata.get("author", ""),
                "pdf_subject": pdf_metadata.get("subject", ""),
                "pdf_creator": pdf_metadata.get("creator", ""),
                "page_count": doc.page_count
            })
            
            # Extract text page by page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Clean up the text
                page_text = self._clean_text(page_text)
                
                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_start": len(text_content),
                    "char_end": len(text_content) + len(page_text)
                })
                
                text_content += page_text + "\n"
            
            doc.close()
            
            # Fallback to PyPDF2 if PyMuPDF fails
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            text_content, pages = await self._extract_with_pypdf2(file_path)
        
        return {
            "text": text_content.strip(),
            "pages": pages,
            "metadata": metadata
        }
    
    async def _extract_with_pypdf2(self, file_path: str) -> tuple:
        """Fallback extraction using PyPDF2"""
        text_content = ""
        pages = []
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                page_text = self._clean_text(page_text)
                
                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_start": len(text_content),
                    "char_end": len(text_content) + len(page_text)
                })
                
                text_content += page_text + "\n"
        
        return text_content.strip(), pages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)  # Space between letters and numbers
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)  # Space between numbers and letters
        
        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip very short lines that might be page numbers
            if len(line) < 3:
                continue
                
            # Skip lines that are just numbers (page numbers)
            if line.isdigit():
                continue
                
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    async def _extract_document_metadata(
        self, 
        text: str, 
        pdf_metadata: Dict[str, Any], 
        extra_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract document metadata from text and PDF metadata"""
        
        # Initialize metadata with defaults
        metadata = {
            "title": "Unknown Title",
            "authors": [],
            "abstract": "",
            "year": None,
            "venue": None,
            "doi": None
        }
        
        # Extract title (from PDF metadata or first meaningful line)
        title = pdf_metadata.get("pdf_title", "").strip()
        if not title:
            # Try to extract from first few lines
            lines = text.split('\n')[:10]
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not line.isupper():
                    title = line
                    break
        
        if title:
            metadata["title"] = self._clean_title(title)
        
        # Extract authors
        authors = self._extract_authors(text, pdf_metadata.get("pdf_author", ""))
        metadata["authors"] = authors
        
        # Extract abstract
        abstract = self._extract_abstract(text)
        metadata["abstract"] = abstract
        
        # Extract year
        year = self._extract_year(text)
        metadata["year"] = year
        
        # Extract DOI
        doi = self._extract_doi(text)
        metadata["doi"] = doi
        
        # Extract venue/conference
        venue = self._extract_venue(text)
        metadata["venue"] = venue
        
        # Override with extra metadata if provided
        metadata.update(extra_metadata)
        
        return metadata
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize title"""
        # Remove common artifacts
        title = re.sub(r'^(Title:|TITLE:)\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Limit length
        if len(title) > 200:
            title = title[:200] + "..."
        
        return title
    
    def _extract_authors(self, text: str, pdf_author: str) -> List[str]:
        """Extract author names"""
        authors = []
        
        # Try PDF metadata first
        if pdf_author:
            authors.extend([name.strip() for name in pdf_author.split(',')])
        
        # Try to extract from text using patterns
        # Look for author patterns in first 1000 characters
        text_start = text[:1000]
        
        # Pattern for academic papers: "Author1, Author2, and Author3"
        author_patterns = [
            r'(?:Authors?|By):\s*([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)\s*(?:\n|\r|\r\n)\s*(?:Abstract|ABSTRACT)'
        ]
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text_start, re.MULTILINE)
            for match in matches:
                author_names = [name.strip() for name in match.split(',')]
                authors.extend(author_names)
                break
        
        # Clean and deduplicate
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if len(author) > 3 and len(author) < 50:  # Basic validation
                cleaned_authors.append(author)
        
        return list(dict.fromkeys(cleaned_authors))  # Remove duplicates while preserving order
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from text"""
        # Look for abstract section
        abstract_patterns = [
            r'(?:Abstract|ABSTRACT)[:\s]*\n?([^{\n]+(?:\n[^{\n]+)*?)(?:\n\s*\n|\nIntroduction|\nKeywords|\n1\.|\n\d+\.)',
            r'(?:Abstract|ABSTRACT)[:\s]*([^{\n]+(?:\n[^{\n]+)*?)(?:\nIntroduction|\nKeywords|\n1\.)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Minimum length check
                    return abstract
        
        return ""
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year"""
        # Look for 4-digit years in first 2000 characters
        text_start = text[:2000]
        year_pattern = r'\b(20\d{2}|19\d{2})\b'
        
        years = re.findall(year_pattern, text_start)
        if years:
            # Return the first reasonable year
            for year in years:
                year_int = int(year)
                if 1990 <= year_int <= datetime.now().year:
                    return year_int
        
        return None
    
    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI if present"""
        doi_pattern = r'(?:doi|DOI):\s*(10\.\d+\/[^\s]+)'
        match = re.search(doi_pattern, text)
        if match:
            return match.group(1)
        return None
    
    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract venue/conference name"""
        # Common conference/journal patterns
        venue_patterns = [
            r'(?:Published in|Proceedings of|Conference on)\s+([A-Z][^{\n]+?)(?:\n|\d{4})',
            r'([A-Z]{2,}[-\s]\d{4})',  # Conference acronyms with years
            r'(Journal of [A-Z][^{\n]+?)(?:\n|Vol\.|Volume)',
        ]
        
        text_start = text[:1000]
        for pattern in venue_patterns:
            match = re.search(pattern, text_start)
            if match:
                venue = match.group(1).strip()
                if len(venue) > 5 and len(venue) < 100:
                    return venue
        
        return None
    
    async def _chunk_document(
        self, 
        text: str, 
        pages: List[Dict[str, Any]], 
        doc_id: str
    ) -> List[DocumentChunk]:
        """Split document into semantic chunks"""
        chunks = []
        
        # First try semantic chunking by sections
        section_chunks = self._chunk_by_sections(text)
        
        if section_chunks:
            # Process section-based chunks
            for i, section_text in enumerate(section_chunks):
                if len(section_text.strip()) < 50:  # Skip very short sections
                    continue
                
                # Further split long sections
                sub_chunks = self._split_long_text(section_text, self.chunk_size, self.overlap)
                
                for j, chunk_text in enumerate(sub_chunks):
                    chunk_id = f"{doc_id}_sec_{i}_chunk_{j}"
                    page_num = self._find_page_number(chunk_text, pages)
                    
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text.strip(),
                        chunk_type="section",
                        page_number=page_num,
                        start_char=0,  # Could be calculated more precisely
                        end_char=len(chunk_text),
                        metadata={"section_index": i, "chunk_index": j}
                    )
                    chunks.append(chunk)
        else:
            # Fallback to sliding window chunking
            sliding_chunks = self._split_long_text(text, self.chunk_size, self.overlap)
            
            for i, chunk_text in enumerate(sliding_chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                page_num = self._find_page_number(chunk_text, pages)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    chunk_type="paragraph",
                    page_number=page_num,
                    start_char=0,
                    end_char=len(chunk_text),
                    metadata={"chunk_index": i}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sections(self, text: str) -> List[str]:
        """Split text by sections/headings"""
        # Look for section headers (numbers, Roman numerals, etc.)
        section_patterns = [
            r'\n\s*(\d+\.?\s+[A-Z][^{\n]+)\n',  # 1. Introduction
            r'\n\s*([IVX]+\.?\s+[A-Z][^{\n]+)\n',  # I. Introduction  
            r'\n\s*([A-Z][A-Z\s]{2,})\n',  # ALL CAPS HEADERS
        ]
        
        sections = []
        last_pos = 0
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            
            if len(matches) > 2:  # Need at least 3 sections to be meaningful
                for i, match in enumerate(matches):
                    if i == 0:
                        # Add intro section
                        intro = text[last_pos:match.start()].strip()
                        if intro:
                            sections.append(intro)
                    
                    # Add section content
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    section_text = text[start:end].strip()
                    if section_text:
                        sections.append(section_text)
                
                return sections
        
        return []  # No clear sections found
    
    def _split_long_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split long text using sliding window with sentence boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
            
            i += 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.nlp:
            # Use spaCy for better sentence segmentation
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= overlap_size:
            return text
        
        # Try to get overlap at sentence boundary
        sentences = self._split_into_sentences(text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= overlap_size:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip()
    
    def _find_page_number(self, chunk_text: str, pages: List[Dict[str, Any]]) -> int:
        """Find which page a chunk belongs to"""
        # Simple heuristic: find page with most overlapping text
        best_page = 1
        best_overlap = 0
        
        chunk_words = set(chunk_text.lower().split())
        
        for page in pages:
            page_words = set(page["text"].lower().split())
            overlap = len(chunk_words.intersection(page_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page["page_number"]
        
        return best_page
    
    def _generate_doc_id(self, title: str) -> str:
        """Generate unique document ID from title"""
        # Create hash from title for uniqueness
        title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
        
        # Create readable ID
        title_clean = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())
        title_words = title_clean.split()[:5]  # First 5 words
        title_slug = '_'.join(title_words)
        
        return f"{title_slug}_{title_hash}"