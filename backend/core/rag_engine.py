"""
RAG Engine - Core retrieval and generation pipeline
Integrates Endee vector database with LLMs for question answering
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import openai
from sklearn.feature_extraction.text import TfidfVectorizer

from .endee_client import EndeeClient, EndeeMetricsCollector
from .document_processor import ProcessedDocument
from utils.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Handles text embeddings using SentenceTransformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dim: {self.dimension})")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to dense embeddings"""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.model.encode([text], convert_to_tensor=False)[0]


class SparseEncoder:
    """Handles sparse vector encoding using TF-IDF"""
    
    def __init__(self, max_features: int = 50000):
        self.max_features = max_features
        self.vectorizer = None
        self.is_fitted = False
    
    def fit(self, documents: List[str]):
        """Fit the sparse encoder on document corpus"""
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.vectorizer.fit(documents)
        self.is_fitted = True
        logger.info(f"Fitted sparse encoder with {len(self.vectorizer.vocabulary_)} features")
    
    def encode(self, texts: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Encode texts to sparse vectors (indices, values)"""
        if not self.is_fitted:
            raise ValueError("Sparse encoder not fitted yet")
        
        sparse_vectors = []
        tfidf_matrix = self.vectorizer.transform(texts)
        
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(i).tocoo()
            indices = row.col.astype(np.uint32)
            values = row.data.astype(np.float32)
            
            # Only keep top-k values for efficiency
            if len(values) > 100:
                top_k_idx = np.argsort(values)[-100:]
                indices = indices[top_k_idx]
                values = values[top_k_idx]
            
            sparse_vectors.append((indices, values))
        
        return sparse_vectors
    
    def encode_single(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode single text to sparse vector"""
        return self.encode([text])[0]


class RAGEngine:
    """Main RAG engine combining retrieval and generation"""
    
    def __init__(self, endee_client: EndeeClient):
        self.endee_client = endee_client
        self.metrics_collector = EndeeMetricsCollector(endee_client)
        
        # Initialize models
        self.embedding_model = EmbeddingModel(settings.embedding_model)
        self.sparse_encoder = SparseEncoder()
        
        # LLM configuration
        if settings.llm_provider == "openai":
            openai.api_key = settings.openai_api_key
        
        # Document storage
        self.documents = {}  # In-memory document store
        
        logger.info("RAG Engine initialized")
    
    async def initialize_index(self):
        """Initialize Endee index and fit sparse encoder if needed"""
        success = await self.endee_client.create_index_if_not_exists()
        if not success:
            raise RuntimeError("Failed to initialize Endee index")
        
        # Check if we need to fit sparse encoder
        index_info = await self.endee_client.get_index_info()
        if index_info and index_info.get("total_elements", 0) == 0:
            logger.info("Empty index - will fit sparse encoder when first documents are added")
        
        logger.info("RAG Engine index initialized")
    
    async def add_document(self, document: ProcessedDocument):
        """Add a processed document to the vector index"""
        try:
            start_time = time.time()
            
            # Store document metadata
            self.documents[document.doc_id] = document
            
            # Prepare texts for embedding
            all_chunks = []
            chunk_metadata = []
            
            for chunk in document.chunks:
                all_chunks.append(chunk.text)
                chunk_metadata.append({
                    "doc_id": document.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "title": document.title,
                    "authors": document.authors,
                    "abstract": document.abstract,
                    "year": document.year,
                    "venue": document.venue,
                    "chunk_type": chunk.chunk_type,
                    "page_number": chunk.page_number
                })
            
            # Fit sparse encoder if not fitted yet
            if not self.sparse_encoder.is_fitted:
                logger.info("Fitting sparse encoder with initial documents...")
                # Get existing documents for fitting
                all_texts = [doc.get_full_text() for doc in self.documents.values()]
                self.sparse_encoder.fit(all_texts)
            
            # Generate embeddings
            dense_embeddings = self.embedding_model.encode(all_chunks)
            sparse_embeddings = self.sparse_encoder.encode(all_chunks)
            
            # Prepare vectors for Endee
            vectors = []
            for i, (chunk, metadata) in enumerate(zip(all_chunks, chunk_metadata)):
                vector_data = {
                    "id": f"{document.doc_id}_{i}",
                    "dense_vector": dense_embeddings[i],
                    "sparse_vector": sparse_embeddings[i],
                    "metadata": metadata
                }
                vectors.append(vector_data)
            
            # Insert into Endee
            success = await self.endee_client.insert_vectors(vectors)
            
            if success:
                processing_time = (time.time() - start_time) * 1000
                logger.info(f"Added document {document.doc_id} with {len(vectors)} chunks in {processing_time:.2f}ms")
                
                # Record metrics
                await self.metrics_collector.record_query(
                    "document_add", processing_time, len(vectors)
                )
            else:
                raise RuntimeError("Failed to insert vectors into Endee")
                
        except Exception as e:
            logger.error(f"Error adding document {document.doc_id}: {e}")
            raise
    
    async def semantic_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using dense embeddings"""
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode_single(query)
            
            # Search in Endee
            results = await self.endee_client.search_vectors(
                query_vector=query_embedding,
                k=k,
                ef=max(k * 2, 100),  # Dynamic ef parameter
                filters=filters
            )
            
            # Enrich results with metadata
            enriched_results = await self._enrich_search_results(results, include_metadata)
            
            search_time = (time.time() - start_time) * 1000
            
            # Record metrics
            await self.metrics_collector.record_query(
                "semantic_search", search_time, len(results)
            )
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining dense and sparse vectors"""
        try:
            start_time = time.time()
            
            # Generate embeddings
            dense_embedding = self.embedding_model.encode_single(query)
            sparse_embedding = self.sparse_encoder.encode_single(query)
            
            # Search in Endee with both dense and sparse
            results = await self.endee_client.search_vectors(
                query_vector=dense_embedding,
                sparse_query=sparse_embedding,
                k=k,
                ef=max(k * 2, 100),
                filters=filters
            )
            
            # Enrich results
            enriched_results = await self._enrich_search_results(results, include_metadata)
            
            search_time = (time.time() - start_time) * 1000
            
            # Record metrics
            await self.metrics_collector.record_query(
                "hybrid_search", search_time, len(results)
            )
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    async def answer_question(
        self,
        question: str,
        context_limit: int = 5000,
        include_citations: bool = True,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Answer question using RAG pipeline"""
        try:
            start_time = time.time()
            
            # Retrieve relevant context
            search_results = await self.hybrid_search(
                query=question,
                k=20,  # Get more candidates for better context
                include_metadata=True
            )
            
            # Prepare context from search results
            context_chunks = []
            citations = []
            current_length = 0
            
            for result in search_results:
                chunk_text = result.get("text", "")
                if current_length + len(chunk_text) > context_limit:
                    break
                
                context_chunks.append(chunk_text)
                current_length += len(chunk_text)
                
                if include_citations:
                    citations.append({
                        "title": result.get("title", ""),
                        "authors": result.get("authors", []),
                        "score": result.get("score", 0.0),
                        "chunk_id": result.get("chunk_id", "")
                    })
            
            # Generate answer using LLM
            context = "\n\n".join(context_chunks)
            answer = await self._generate_answer(question, context, temperature)
            
            response_time = (time.time() - start_time) * 1000
            
            # Calculate confidence based on top search scores
            confidence = np.mean([r.get("score", 0) for r in search_results[:5]]) if search_results else 0.0
            
            # Record metrics
            await self.metrics_collector.record_query(
                "question_answering", response_time, len(search_results)
            )
            
            return {
                "answer": answer,
                "citations": citations if include_citations else [],
                "confidence": float(confidence),
                "response_time_ms": response_time,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise
    
    async def _generate_answer(self, question: str, context: str, temperature: float) -> str:
        """Generate answer using LLM"""
        try:
            system_prompt = """You are a helpful AI assistant that answers questions based on provided scientific literature context. 
            
Guidelines:
- Answer based only on the provided context
- Be precise and cite specific information when possible
- If the context doesn't contain enough information, say so
- Use clear, academic language
- Provide detailed explanations when appropriate"""

            user_prompt = f"""Context from scientific papers:
{context}

Question: {question}

Answer:"""

            if settings.llm_provider == "openai":
                response = await openai.ChatCompletion.acreate(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
            
            else:
                # Fallback to simple context-based response
                return f"Based on the provided context, here is a summary relevant to your question: {context[:500]}..."
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."
    
    async def _enrich_search_results(
        self, 
        results: List[Dict[str, Any]], 
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Enrich search results with document metadata and text"""
        enriched = []
        
        for result in results:
            chunk_id = result.get("id", "")
            doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
            
            enriched_result = {
                "id": result.get("id"),
                "score": result.get("score", 0.0),
                "distance": result.get("distance", 0.0)
            }
            
            if include_metadata and doc_id in self.documents:
                doc = self.documents[doc_id]
                
                # Find the specific chunk
                chunk_idx = int(chunk_id.split("_")[-1]) if "_" in chunk_id else 0
                if chunk_idx < len(doc.chunks):
                    chunk = doc.chunks[chunk_idx]
                    
                    enriched_result.update({
                        "text": chunk.text,
                        "title": doc.title,
                        "authors": doc.authors,
                        "abstract": doc.abstract,
                        "year": doc.year,
                        "venue": doc.venue,
                        "chunk_type": chunk.chunk_type,
                        "page_number": chunk.page_number,
                        "chunk_id": chunk.chunk_id
                    })
            
            enriched.append(enriched_result)
        
        return enriched
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if doc_id in self.documents:
            doc = self.documents[doc_id]
            return {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "authors": doc.authors,
                "abstract": doc.abstract,
                "year": doc.year,
                "venue": doc.venue,
                "chunk_count": len(doc.chunks),
                "full_text_length": len(doc.get_full_text())
            }
        return None
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from index and storage"""
        try:
            # Delete from vector index using filter
            deleted_count = await self.endee_client.delete_vectors_by_filter({
                "doc_id": {"$eq": doc_id}
            })
            
            # Remove from local storage
            if doc_id in self.documents:
                del self.documents[doc_id]
            
            logger.info(f"Deleted document {doc_id}, removed {deleted_count} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return await self.metrics_collector.get_metrics()
    
    async def get_query_analytics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query analytics"""
        return self.metrics_collector._query_history[-limit:]
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        index_info = await self.endee_client.get_index_info()
        
        return {
            "index_info": index_info,
            "document_count": len(self.documents),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.dimension,
            "sparse_encoder_fitted": self.sparse_encoder.is_fitted
        }
    
    async def rebuild_index(self):
        """Rebuild the entire index (for maintenance)"""
        logger.info("Starting index rebuild...")
        
        # This would involve recreating the index and re-adding all documents
        # Implementation depends on specific requirements
        pass
    
    async def shutdown(self):
        """Cleanup resources"""
        await self.endee_client.close()
        logger.info("RAG Engine shutdown complete")