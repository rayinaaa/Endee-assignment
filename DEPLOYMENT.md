# Deployment Guide

This guide covers different deployment scenarios for the Scientific RAG System.

## Quick Start with Docker Compose

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available
- OpenAI API key (optional, for full QA features)

### Steps

1. **Clone and prepare the repository**
```bash
git clone https://github.com/your-username/scientific-rag-endee
cd scientific-rag-endee
```

2. **Set environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

3. **Start the system**
```bash
docker-compose up -d
```

4. **Verify deployment**
```bash
# Check service health
docker-compose ps

# Test API
curl http://localhost:8000/health

# Access frontend
open http://localhost:3000
```

## Manual Development Setup

### Backend Setup

1. **Install Python dependencies**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Start the backend**
```bash
export ENDEE_HOST=localhost
export ENDEE_PORT=8080
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Install Node.js dependencies**
```bash
cd frontend
npm install
```

2. **Start the frontend**
```bash
export REACT_APP_API_URL=http://localhost:8000
npm start
```

### Endee Setup

1. **Build Endee**
```bash
cd endee-labs
./install.sh --release --avx2  # or --neon for ARM
```

2. **Start Endee**
```bash
./build/ndd
```

## Production Deployment

### Docker Compose with Nginx

```bash
# Start with production profile
docker-compose --profile production up -d
```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests:

```bash
kubectl apply -f k8s/
```

### Cloud Deployment Options

#### AWS ECS with Fargate
- Use the provided task definitions in `aws/ecs-task-definitions/`
- Configure Application Load Balancer
- Set up CloudWatch logging

#### Google Cloud Run
- Build and push containers to GCR
- Deploy using Cloud Run services
- Configure load balancing and SSL

#### Azure Container Instances
- Use Azure Container Registry
- Deploy container groups
- Configure Application Gateway

## Configuration

### Environment Variables

#### Backend Configuration
```bash
# Endee connection
ENDEE_HOST=localhost
ENDEE_PORT=8080

# LLM configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-3.5-turbo

# Document processing
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### Endee Configuration
```bash
# Data directory
NDD_DATA_DIR=/app/data

# Server settings
NDD_SERVER_PORT=8080
NDD_MAX_MEMORY_GB=4

# Authentication (leave empty to disable)
NDD_AUTH_TOKEN=""
```

## Scaling and Performance

### Horizontal Scaling

1. **Multiple Backend Replicas**
   - Use load balancer to distribute requests
   - Ensure shared storage for uploaded documents
   - Configure session affinity if needed

2. **Endee Clustering**
   - Deploy multiple Endee instances
   - Use consistent hashing for data distribution
   - Implement read replicas for better query performance

### Performance Tuning

1. **Endee Optimization**
   - Adjust `M` parameter for index connectivity
   - Tune `ef_construction` for build vs search trade-off
   - Use appropriate quantization level (INT8 recommended)

2. **Backend Optimization**
   - Use connection pooling
   - Implement caching for frequent queries
   - Optimize chunk size based on use case

3. **Frontend Optimization**
   - Enable CDN for static assets
   - Implement query debouncing
   - Use virtual scrolling for large result sets

## Monitoring and Logging

### Health Checks

All services include health check endpoints:
- Backend: `GET /health`
- Endee: `GET /api/v1/health` 
- Frontend: HTTP 200 on root path

### Metrics Collection

1. **Application Metrics**
   - Query latency and throughput
   - Index size and growth
   - Error rates and types

2. **Infrastructure Metrics**
   - CPU and memory usage
   - Disk I/O and storage
   - Network traffic

### Logging Configuration

```yaml
# docker-compose.override.yml
services:
  backend:
    environment:
      - LOG_LEVEL=DEBUG
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
```

## Security

### Authentication and Authorization

1. **API Authentication**
   - Use API keys or JWT tokens
   - Implement rate limiting
   - Enable CORS properly

2. **Endee Security**
   - Set authentication token in production
   - Use network policies to restrict access
   - Enable TLS for inter-service communication

### Data Protection

1. **Data at Rest**
   - Encrypt volumes in cloud deployments
   - Use managed encryption keys
   - Regular backup procedures

2. **Data in Transit**
   - TLS termination at load balancer
   - Service mesh for internal encryption
   - Secure API endpoints

## Backup and Recovery

### Data Backup

1. **Endee Data**
   - Use built-in backup functionality
   - Schedule regular snapshots
   - Store backups in different region/zone

2. **Document Storage**
   - Backup uploaded documents
   - Maintain metadata consistency
   - Test recovery procedures

### Disaster Recovery

1. **Multi-region Setup**
   - Deploy in multiple availability zones
   - Use geo-replicated storage
   - Implement automated failover

2. **Recovery Procedures**
   - Document step-by-step recovery
   - Test recovery regularly
   - Monitor RTO and RPO metrics

## Troubleshooting

### Common Issues

1. **Endee Connection Failed**
   - Check network connectivity
   - Verify port configuration
   - Review Endee logs

2. **High Memory Usage**
   - Monitor index size
   - Adjust quantization settings
   - Configure memory limits

3. **Slow Search Performance**
   - Tune HNSW parameters
   - Check index optimization
   - Monitor system resources

### Debug Commands

```bash
# Check service logs
docker-compose logs -f backend
docker-compose logs -f endee

# Test API endpoints
curl -X POST http://localhost:8000/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 5}'

# Monitor resource usage
docker stats

# Check index statistics
curl http://localhost:8080/api/v1/index/scientific_papers/info
```

## Performance Benchmarks

### Expected Performance

| Metric | Development | Production |
|--------|-------------|------------|
| Query Latency (p95) | < 100ms | < 50ms |
| Throughput | 100 QPS | 1000+ QPS |
| Index Size | 100K docs | 1M+ docs |
| Memory Usage | 2-4GB | 8-16GB |

### Load Testing

Use the provided load testing scripts:

```bash
# Install dependencies
pip install locust

# Run load tests
cd tests/load
locust -f search_load_test.py --host=http://localhost:8000
```

## Maintenance

### Regular Tasks

1. **Index Optimization**
   - Monitor index fragmentation
   - Rebuild if necessary
   - Update embedding models

2. **System Updates**
   - Update dependencies
   - Apply security patches
   - Test in staging first

3. **Performance Review**
   - Analyze query patterns
   - Optimize slow queries
   - Review resource allocation