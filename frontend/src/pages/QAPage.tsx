import React, { useState } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Rating,
} from '@mui/material';
import { QuestionAnswer, ExpandMore } from '@mui/icons-material';
import { askQuestion } from '../services/api';

interface Citation {
  title: string;
  authors: string[];
  score: number;
  chunk_id: string;
}

interface QAResult {
  answer: string;
  citations: Citation[];
  confidence: number;
  response_time_ms: number;
}

const QAPage: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<QAResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAskQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await askQuestion({
        question: question.trim(),
        context_limit: 5000,
        include_citations: true,
        temperature: 0.1,
      });

      setResult(response);
    } catch (err) {
      setError('Failed to answer question. Please try again.');
      console.error('QA error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskQuestion();
    }
  };

  const exampleQuestions = [
    "What are the main challenges in transformer interpretability?",
    "How does BERT handle attention mechanisms?",
    "What are the differences between supervised and unsupervised learning?",
    "Explain the concept of transfer learning in deep neural networks",
  ];

  return (
    <Box sx={{ maxWidth: 1000, mx: 'auto', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Question Answering
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Ask questions about scientific literature and get detailed answers with citations
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <TextField
          fullWidth
          label="Ask a Question"
          variant="outlined"
          multiline
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="e.g., What are the main challenges in transformer interpretability?"
          disabled={loading}
          sx={{ mb: 2 }}
        />
        
        <Button
          variant="contained"
          startIcon={loading ? <CircularProgress size={20} /> : <QuestionAnswer />}
          onClick={handleAskQuestion}
          disabled={loading || !question.trim()}
          sx={{ mb: 2 }}
        >
          {loading ? 'Answering...' : 'Ask Question'}
        </Button>

        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Example questions:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {exampleQuestions.map((example, index) => (
              <Chip
                key={index}
                label={example}
                variant="outlined"
                size="small"
                onClick={() => setQuestion(example)}
                sx={{ cursor: 'pointer' }}
              />
            ))}
          </Box>
        </Box>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'flex-start', mb: 2 }}>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                  Answer
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    Confidence:
                  </Typography>
                  <Rating
                    value={result.confidence * 5}
                    precision={0.1}
                    size="small"
                    readOnly
                  />
                  <Typography variant="body2" color="text.secondary">
                    ({Math.round(result.confidence * 100)}%)
                  </Typography>
                </Box>
              </Box>
              
              <Typography variant="body1" sx={{ lineHeight: 1.7, mb: 2 }}>
                {result.answer}
              </Typography>
              
              <Typography variant="caption" color="text.secondary">
                Response time: {result.response_time_ms.toFixed(0)}ms
              </Typography>
            </CardContent>
          </Card>

          {result.citations.length > 0 && (
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">
                  Sources & Citations ({result.citations.length})
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {result.citations.map((citation, index) => (
                    <Card key={index} variant="outlined">
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'flex-start', mb: 1 }}>
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            {citation.title}
                          </Typography>
                          <Chip
                            label={`Relevance: ${citation.score.toFixed(3)}`}
                            size="small"
                            color="primary"
                            variant="outlined"
                          />
                        </Box>
                        
                        {citation.authors.length > 0 && (
                          <Typography variant="body2" color="text.secondary">
                            Authors: {citation.authors.join(', ')}
                          </Typography>
                        )}
                        
                        <Typography variant="caption" color="text.secondary">
                          Chunk ID: {citation.chunk_id}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              </AccordionDetails>
            </Accordion>
          )}
        </Box>
      )}

      {!result && !loading && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h6" color="text.secondary">
            Ask a question to get started
          </Typography>
          <Typography variant="body2" color="text.secondary">
            The system will search through scientific literature to provide detailed answers
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default QAPage;