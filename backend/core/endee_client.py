"""
Endee Vector Database Client
High-performance client for interacting with Endee
"""

import json
import httpx
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class EndeeClient:
    """Async client for Endee vector database"""
    
    def __init__(self, host: str = "localhost", port: int = 8080, timeout: int = 30):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Index configuration
        self.index_name = "scientific_papers"
        self.dense_dim = 768  # SentenceTransformers dimension
        self.sparse_dim = 50000  # Sparse vocabulary size
        
    async def health_check(self) -> bool:
        """Check if Endee server is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def create_index_if_not_exists(self) -> bool:
        """Create the main index if it doesn't exist"""
        try:
            # Check if index exists
            response = await self.client.get(f"{self.base_url}/api/v1/index/{self.index_name}/info")
            
            if response.status_code == 200:
                logger.info(f"Index '{self.index_name}' already exists")
                return True
            
            # Create new index with hybrid support
            index_config = {
                "index_name": self.index_name,
                "dim": self.dense_dim,
                "sparse_dim": self.sparse_dim,
                "space_type": "cosine",
                "M": 32,  # HNSW parameter for connectivity
                "ef_con": 200,  # HNSW construction parameter
                "precision": "INT8",  # Use quantization for efficiency
                "size_in_millions": 1  # Support up to 1M documents
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/index/create",
                json=index_config
            )
            
            if response.status_code == 200:
                logger.info(f"Created index '{self.index_name}' successfully")
                return True
            else:
                logger.error(f"Failed to create index: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    async def insert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Insert vectors into the index"""
        try:
            # Prepare vectors for Endee format
            endee_vectors = []
            for vector in vectors:
                endee_vector = {
                    "id": vector["id"],
                    "vector": vector["dense_vector"].tolist() if isinstance(vector["dense_vector"], np.ndarray) else vector["dense_vector"]
                }
                
                # Add sparse vectors if available
                if "sparse_vector" in vector and vector["sparse_vector"] is not None:
                    sparse_indices, sparse_values = vector["sparse_vector"]
                    endee_vector["sparse_indices"] = sparse_indices.tolist() if isinstance(sparse_indices, np.ndarray) else sparse_indices
                    endee_vector["sparse_values"] = sparse_values.tolist() if isinstance(sparse_values, np.ndarray) else sparse_values
                
                endee_vectors.append(endee_vector)
            
            # Insert vectors
            response = await self.client.post(
                f"{self.base_url}/api/v1/index/{self.index_name}/vector/insert",
                json=endee_vectors,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Inserted {len(vectors)} vectors successfully")
                return True
            else:
                logger.error(f"Failed to insert vectors: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            return False
    
    async def search_vectors(
        self, 
        query_vector: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        sparse_query: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            search_request = {
                "k": k,
                "vector": query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                "include_vectors": False
            }
            
            # Add sparse query if provided
            if sparse_query is not None:
                sparse_indices, sparse_values = sparse_query
                search_request["sparse_indices"] = sparse_indices.tolist() if isinstance(sparse_indices, np.ndarray) else sparse_indices
                search_request["sparse_values"] = sparse_values.tolist() if isinstance(sparse_values, np.ndarray) else sparse_values
            
            # Add ef parameter for search quality
            if ef is not None:
                search_request["ef"] = ef
            
            # Add filters if provided
            if filters:
                search_request["filter"] = json.dumps([filters])
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/index/{self.index_name}/search",
                json=search_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                # Handle MessagePack response
                content_type = response.headers.get("content-type", "")
                if "msgpack" in content_type:
                    import msgpack
                    results = msgpack.unpackb(response.content, raw=False)
                else:
                    results = response.json()
                
                return self._format_search_results(results)
            else:
                logger.error(f"Search failed: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def _format_search_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format Endee search results"""
        formatted_results = []
        
        if "ids" in raw_results and "distances" in raw_results:
            ids = raw_results["ids"]
            distances = raw_results["distances"]
            
            for doc_id, distance in zip(ids, distances):
                # Convert distance to similarity score (cosine similarity)
                similarity = 1.0 - distance
                
                formatted_results.append({
                    "id": str(doc_id),
                    "score": float(similarity),
                    "distance": float(distance)
                })
        
        return formatted_results
    
    async def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific vector by ID"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/index/{self.index_name}/vector/get",
                json={"id": vector_id}
            )
            
            if response.status_code == 200:
                # Handle MessagePack response
                content_type = response.headers.get("content-type", "")
                if "msgpack" in content_type:
                    import msgpack
                    result = msgpack.unpackb(response.content, raw=False)
                else:
                    result = response.json()
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting vector {vector_id}: {e}")
            return None
    
    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID"""
        try:
            response = await self.client.delete(
                f"{self.base_url}/api/v1/index/{self.index_name}/vector/{vector_id}/delete"
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error deleting vector {vector_id}: {e}")
            return False
    
    async def get_index_info(self) -> Optional[Dict[str, Any]]:
        """Get index information and statistics"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/index/{self.index_name}/info"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return None
    
    async def update_filters(self, updates: List[Dict[str, Any]]) -> bool:
        """Update filters for multiple vectors"""
        try:
            filter_updates = []
            for update in updates:
                filter_updates.append({
                    "id": update["id"],
                    "filter": json.dumps(update["filter"])
                })
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/index/{self.index_name}/filters/update",
                json={"updates": filter_updates}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error updating filters: {e}")
            return False
    
    async def delete_vectors_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching a filter"""
        try:
            response = await self.client.delete(
                f"{self.base_url}/api/v1/index/{self.index_name}/vectors/delete",
                json={"filter": [filters]}
            )
            
            if response.status_code == 200:
                # Parse response to get count
                result_text = response.text
                # Extract number from response like "123 vectors deleted"
                import re
                match = re.search(r'(\d+)', result_text)
                return int(match.group(1)) if match else 0
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting vectors by filter: {e}")
            return 0
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class EndeeMetricsCollector:
    """Collect and aggregate metrics from Endee"""
    
    def __init__(self, client: EndeeClient):
        self.client = client
        self._metrics = {}
        self._query_history = []
    
    async def record_query(self, query_type: str, latency_ms: float, result_count: int):
        """Record query metrics"""
        self._query_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": query_type,
            "latency_ms": latency_ms,
            "result_count": result_count
        })
        
        # Keep only recent history (last 1000 queries)
        if len(self._query_history) > 1000:
            self._query_history = self._query_history[-1000:]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        index_info = await self.client.get_index_info()
        
        # Calculate query metrics
        recent_queries = self._query_history[-100:] if self._query_history else []
        avg_latency = sum(q["latency_ms"] for q in recent_queries) / max(len(recent_queries), 1)
        
        return {
            "index_stats": index_info or {},
            "query_metrics": {
                "total_queries": len(self._query_history),
                "recent_avg_latency_ms": avg_latency,
                "recent_query_count": len(recent_queries)
            },
            "system_metrics": {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy"
            }
        }