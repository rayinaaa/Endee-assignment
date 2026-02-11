import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface SearchRequest {
  query: string;
  k: number;
  hybrid?: boolean;
  filters?: Record<string, any>;
}

export interface SearchResponse {
  query: string;
  results: Array<{
    id: string;
    score: number;
    text?: string;
    title?: string;
    authors?: string[];
    abstract?: string;
    year?: number;
    venue?: string;
    page_number?: number;
  }>;
  total_found: number;
  search_time_ms: number;
}

export interface QARequest {
  question: string;
  context_limit?: number;
  include_citations?: boolean;
  temperature?: number;
}

export interface QAResponse {
  question: string;
  answer: string;
  citations: Array<{
    title: string;
    authors: string[];
    score: number;
    chunk_id: string;
  }>;
  confidence: number;
  response_time_ms: number;
}

export const performSearch = async (request: SearchRequest): Promise<SearchResponse> => {
  const endpoint = request.hybrid ? '/search/hybrid' : '/search/semantic';
  const response = await apiClient.post(endpoint, request);
  return response.data;
};

export const askQuestion = async (request: QARequest): Promise<QAResponse> => {
  const response = await apiClient.post('/qa/answer', request);
  return response.data;
};

export const uploadDocument = async (file: File, metadata?: Record<string, any>) => {
  const formData = new FormData();
  formData.append('file', file);
  if (metadata) {
    formData.append('metadata', JSON.stringify(metadata));
  }

  const response = await apiClient.post('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getSystemMetrics = async () => {
  const response = await apiClient.get('/analytics/metrics');
  return response.data;
};

export const getQueryAnalytics = async (limit: number = 100) => {
  const response = await apiClient.get(`/analytics/queries?limit=${limit}`);
  return response.data;
};

export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};