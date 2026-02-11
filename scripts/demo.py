#!/usr/bin/env python3
"""
Scientific RAG System Demo Script
Demonstrates key functionality of the system
"""

import asyncio
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
DEMO_QUERIES = [
    "What are the main challenges in transformer interpretability?",
    "How does attention mechanism work in neural networks?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of transfer learning",
    "What are the advantages of using vector databases for AI applications?"
]

DEMO_SEARCH_QUERIES = [
    "machine learning interpretability",
    "neural network optimization",
    "natural language processing transformers",
    "computer vision deep learning",
    "reinforcement learning algorithms"
]

class RAGSystemDemo:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = requests.Session()
        
    def check_health(self) -> bool:
        """Check if the system is healthy"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def demo_semantic_search(self):
        """Demonstrate semantic search functionality"""
        print("\n🔍 SEMANTIC SEARCH DEMO")
        print("=" * 50)
        
        for i, query in enumerate(DEMO_SEARCH_QUERIES, 1):
            print(f"\n{i}. Searching for: '{query}'")
            
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.api_url}/search/semantic",
                    json={
                        "query": query,
                        "k": 5,
                        "include_metadata": True
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    search_time = time.time() - start_time
                    
                    print(f"   ✅ Found {len(data['results'])} results in {search_time*1000:.1f}ms")
                    
                    # Show top result
                    if data['results']:
                        top_result = data['results'][0]
                        print(f"   📄 Top result: {top_result.get('title', 'Untitled')[:60]}...")
                        print(f"   📊 Score: {top_result['score']:.3f}")
                else:
                    print(f"   ❌ Search failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            time.sleep(1)  # Rate limiting
    
    def demo_hybrid_search(self):
        """Demonstrate hybrid search functionality"""
        print("\n🎯 HYBRID SEARCH DEMO")
        print("=" * 50)
        
        query = "transformer attention mechanism"
        print(f"Comparing Dense vs Hybrid search for: '{query}'")
        
        # Dense search
        try:
            response = self.session.post(
                f"{self.api_url}/search/semantic",
                json={"query": query, "k": 3}
            )
            if response.status_code == 200:
                dense_results = response.json()['results']
                print(f"\n📋 Dense Search Results ({len(dense_results)} results):")
                for i, result in enumerate(dense_results, 1):
                    print(f"   {i}. Score: {result['score']:.3f} - {result.get('title', 'Untitled')[:50]}...")
        except Exception as e:
            print(f"❌ Dense search error: {e}")
        
        # Hybrid search
        try:
            response = self.session.post(
                f"{self.api_url}/search/hybrid",
                json={"query": query, "k": 3, "alpha": 0.7}
            )
            if response.status_code == 200:
                hybrid_results = response.json()['results']
                print(f"\n🎯 Hybrid Search Results ({len(hybrid_results)} results):")
                for i, result in enumerate(hybrid_results, 1):
                    print(f"   {i}. Score: {result['score']:.3f} - {result.get('title', 'Untitled')[:50]}...")
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
    
    def demo_question_answering(self):
        """Demonstrate question answering functionality"""
        print("\n❓ QUESTION ANSWERING DEMO")
        print("=" * 50)
        
        for i, question in enumerate(DEMO_QUERIES, 1):
            print(f"\n{i}. Question: {question}")
            print("   " + "-" * 40)
            
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.api_url}/qa/answer",
                    json={
                        "question": question,
                        "context_limit": 3000,
                        "include_citations": True,
                        "temperature": 0.1
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_time = time.time() - start_time
                    
                    print(f"   ✅ Answer generated in {response_time:.1f}s")
                    print(f"   🎯 Confidence: {data['confidence']:.2%}")
                    print(f"   📚 Citations: {len(data['citations'])}")
                    
                    # Show answer (truncated)
                    answer = data['answer']
                    if len(answer) > 200:
                        answer = answer[:200] + "..."
                    print(f"\n   💡 Answer: {answer}")
                    
                    # Show top citation
                    if data['citations']:
                        top_citation = data['citations'][0]
                        print(f"   📖 Top source: {top_citation['title'][:50]}...")
                        
                else:
                    print(f"   ❌ QA failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()  # Add spacing
            time.sleep(2)  # Rate limiting for demo
    
    def demo_system_metrics(self):
        """Show system metrics and statistics"""
        print("\n📊 SYSTEM METRICS DEMO")
        print("=" * 50)
        
        try:
            # Get system metrics
            response = self.session.get(f"{self.api_url}/analytics/metrics")
            if response.status_code == 200:
                metrics = response.json()
                
                print("🏛️  Index Statistics:")
                index_stats = metrics.get('index_stats', {})
                print(f"   📄 Total documents: {index_stats.get('total_elements', 'N/A')}")
                print(f"   📏 Vector dimension: {index_stats.get('dimension', 'N/A')}")
                print(f"   🎛️  Space type: {index_stats.get('space_type', 'N/A')}")
                print(f"   🔧 Precision: {index_stats.get('precision', 'N/A')}")
                
                print("\n⚡ Query Performance:")
                query_metrics = metrics.get('query_metrics', {})
                print(f"   📈 Total queries: {query_metrics.get('total_queries', 'N/A')}")
                print(f"   ⏱️  Avg latency: {query_metrics.get('recent_avg_latency_ms', 'N/A')}ms")
                
            else:
                print(f"❌ Failed to get metrics: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Metrics error: {e}")
    
    def demo_document_upload(self):
        """Demonstrate document upload (if sample file exists)"""
        print("\n📤 DOCUMENT UPLOAD DEMO")
        print("=" * 50)
        
        # Look for sample PDF in project
        sample_files = list(Path(".").glob("**/*.pdf"))
        
        if not sample_files:
            print("ℹ️  No sample PDF files found for upload demo")
            print("   To test upload functionality:")
            print("   1. Add a PDF file to the project directory")
            print("   2. Use the web interface at http://localhost:3000/upload")
            print("   3. Or use the API directly:")
            print("   curl -X POST -F 'file=@sample.pdf' http://localhost:8000/documents/upload")
            return
        
        sample_file = sample_files[0]
        print(f"📁 Found sample file: {sample_file}")
        
        try:
            with open(sample_file, 'rb') as f:
                files = {'file': f}
                response = self.session.post(
                    f"{self.api_url}/documents/upload",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Upload successful: {data['message']}")
                print(f"📄 File: {data['filename']}")
                print(f"🔄 Status: {data['status']}")
            else:
                print(f"❌ Upload failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
    
    def run_full_demo(self):
        """Run the complete demonstration"""
        print("🚀 SCIENTIFIC RAG SYSTEM DEMO")
        print("=" * 60)
        print("Demonstrating Endee-powered RAG capabilities")
        print()
        
        # Health check first
        print("🏥 Checking system health...")
        if not self.check_health():
            print("❌ System is not healthy. Please start the services first:")
            print("   docker-compose up -d")
            return
        
        print("✅ System is healthy!")
        
        # Run demos
        self.demo_system_metrics()
        self.demo_semantic_search()
        self.demo_hybrid_search()
        self.demo_question_answering()
        self.demo_document_upload()
        
        # Final summary
        print("\n🎉 DEMO COMPLETE!")
        print("=" * 60)
        print("✅ Successfully demonstrated:")
        print("   • Semantic search with vector similarity")
        print("   • Hybrid search combining dense and sparse vectors") 
        print("   • Question answering with citation extraction")
        print("   • System metrics and performance monitoring")
        print("   • Document upload and processing capabilities")
        print()
        print("🌐 Access the web interface at: http://localhost:3000")
        print("📚 API documentation at: http://localhost:8000/docs")
        print()
        print("🔧 Powered by:")
        print("   • Endee Vector Database (HNSW + Quantization)")
        print("   • SentenceTransformers (Dense Embeddings)")
        print("   • TF-IDF (Sparse Vectors)")
        print("   • FastAPI (Backend)")
        print("   • React (Frontend)")

def main():
    """Main demo function"""
    demo = RAGSystemDemo(API_BASE_URL)
    demo.run_full_demo()

if __name__ == "__main__":
    main()