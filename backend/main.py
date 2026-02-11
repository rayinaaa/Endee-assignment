"""
Scientific Literature RAG System - Main API Server
Powered by Endee Vector Database
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from core.rag_engine import RAGEngine
from core.document_processor import DocumentProcessor
from core.endee_client import EndeeClient
from models.schemas import (
    SearchRequest, SearchResponse, 
    QARequest, QAResponse,
    DocumentUploadResponse, DocumentMetadata,
    SystemMetrics
)
from utils.config import settings
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
rag_engine: Optional[RAGEngine] = None
document_processor: Optional[DocumentProcessor] = None
endee_client: Optional[EndeeClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global rag_engine, document_processor, endee_client
    
    logger.info("Starting Scientific RAG System...")
    
    # Initialize components
    endee_client = EndeeClient(
        host=settings.endee_host,
        port=settings.endee_port
    )
    
    document_processor = DocumentProcessor()
    rag_engine = RAGEngine(endee_client)
    
    # Initialize Endee index if needed
    await rag_engine.initialize_index()
    
    logger.info("System initialized successfully!")
    
    yield
    
    logger.info("Shutting down...")
    if rag_engine:
        await rag_engine.shutdown()

# Create FastAPI app
app = FastAPI(
    title="Scientific Literature RAG System",
    description="Advanced RAG system using Endee Vector Database for scientific literature",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Endee connection
        endee_status = await endee_client.health_check() if endee_client else False
        
        return {
            "status": "healthy" if endee_status else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "endee_connected": endee_status,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Search endpoints
@app.post("/search/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Perform semantic search across scientific literature"""
    try:
        logger.info(f"Semantic search query: {request.query}")
        
        results = await rag_engine.semantic_search(
            query=request.query,
            k=request.k,
            filters=request.filters,
            include_metadata=request.include_metadata
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            search_time_ms=results[0].get("search_time_ms", 0) if results else 0
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """Perform hybrid search (dense + sparse vectors)"""
    try:
        logger.info(f"Hybrid search query: {request.query}")
        
        results = await rag_engine.hybrid_search(
            query=request.query,
            k=request.k,
            filters=request.filters,
            include_metadata=request.include_metadata,
            alpha=request.alpha  # Weight between dense and sparse
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            search_time_ms=results[0].get("search_time_ms", 0) if results else 0
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Question Answering endpoints
@app.post("/qa/answer", response_model=QAResponse)
async def answer_question(request: QARequest):
    """Answer questions using RAG pipeline"""
    try:
        logger.info(f"QA request: {request.question}")
        
        answer_data = await rag_engine.answer_question(
            question=request.question,
            context_limit=request.context_limit,
            include_citations=request.include_citations,
            temperature=request.temperature
        )
        
        return QAResponse(
            question=request.question,
            answer=answer_data["answer"],
            citations=answer_data.get("citations", []),
            confidence=answer_data.get("confidence", 0.0),
            response_time_ms=answer_data.get("response_time_ms", 0)
        )
        
    except Exception as e:
        logger.error(f"QA failed: {e}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

# Document management endpoints
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload and process a scientific document"""
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_path,
            file.filename,
            metadata
        )
        
        return DocumentUploadResponse(
            filename=file.filename,
            status="processing",
            message="Document uploaded successfully and is being processed"
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(file_path: str, filename: str, metadata: Optional[str]):
    """Background task to process uploaded documents"""
    try:
        # Process the document
        processed_doc = await document_processor.process_pdf(file_path, metadata)
        
        # Add to vector index
        await rag_engine.add_document(processed_doc)
        
        # Clean up temporary file
        os.remove(file_path)
        
        logger.info(f"Document processed successfully: {filename}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {filename}: {e}")

@app.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    """Get document metadata and content"""
    try:
        document = await rag_engine.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except Exception as e:
        logger.error(f"Get document failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the index"""
    try:
        success = await rag_engine.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# Analytics endpoints
@app.get("/analytics/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        metrics = await rag_engine.get_metrics()
        return SystemMetrics(**metrics)
        
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/analytics/queries")
async def get_query_analytics(limit: int = 100):
    """Get recent query analytics"""
    try:
        analytics = await rag_engine.get_query_analytics(limit)
        return {"queries": analytics}
        
    except Exception as e:
        logger.error(f"Get analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# Index management endpoints
@app.post("/admin/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild the vector index (admin only)"""
    try:
        background_tasks.add_task(rag_engine.rebuild_index)
        return {"message": "Index rebuild started"}
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")

@app.get("/admin/index/stats")
async def get_index_stats():
    """Get index statistics"""
    try:
        stats = await rag_engine.get_index_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Get index stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )