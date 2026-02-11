"""
Pydantic models for API request/response schemas
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Search related models
class SearchRequest(BaseModel):
    """Request model for search endpoints"""
    query: str = Field(..., description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    include_metadata: bool = Field(default=True, description="Include document metadata")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Dense/sparse weight for hybrid search")

class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    score: float
    text: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    chunk_type: Optional[str] = None
    page_number: Optional[int] = None

class SearchResponse(BaseModel):
    """Response model for search endpoints"""
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float

# Question Answering models
class QARequest(BaseModel):
    """Request model for question answering"""
    question: str = Field(..., description="Question to answer")
    context_limit: int = Field(default=5000, ge=1000, le=10000, description="Maximum context length")
    include_citations: bool = Field(default=True, description="Include source citations")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")

class Citation(BaseModel):
    """Citation information"""
    title: str
    authors: List[str]
    score: float
    chunk_id: str

class QAResponse(BaseModel):
    """Response model for question answering"""
    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    response_time_ms: float

# Document management models
class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    filename: str
    status: str
    message: str

class DocumentMetadata(BaseModel):
    """Document metadata"""
    doc_id: str
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    venue: Optional[str] = None
    chunk_count: int
    full_text_length: int

# Analytics models
class SystemMetrics(BaseModel):
    """System performance metrics"""
    index_stats: Dict[str, Any]
    query_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]