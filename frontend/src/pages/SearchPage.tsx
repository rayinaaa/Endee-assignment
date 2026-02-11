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
  FormControlLabel,
  Switch,
  Slider,
  Grid,
} from '@mui/material';
import { Search } from '@mui/icons-material';
import { performSearch } from '../services/api';

interface SearchResult {
  id: string;
  score: number;
  text?: string;
  title?: string;
  authors?: string[];
  abstract?: string;
  year?: number;
  venue?: string;
  page_number?: number;
}

const SearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTime, setSearchTime] = useState<number>(0);
  const [useHybrid, setUseHybrid] = useState(true);
  const [maxResults, setMaxResults] = useState(10);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await performSearch({
        query: query.trim(),
        k: maxResults,
        hybrid: useHybrid,
      });

      setResults(response.results);
      setSearchTime(response.search_time_ms);
    } catch (err) {
      setError('Search failed. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Semantic Search
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Search through scientific literature using natural language queries
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              label="Search Query"
              variant="outlined"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="e.g., machine learning interpretability methods"
              disabled={loading}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Button
              fullWidth
              variant="contained"
              startIcon={loading ? <CircularProgress size={20} /> : <Search />}
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              sx={{ height: 56 }}
            >
              {loading ? 'Searching...' : 'Search'}
            </Button>
          </Grid>
        </Grid>

        <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={useHybrid}
                onChange={(e) => setUseHybrid(e.target.checked)}
                disabled={loading}
              />
            }
            label="Hybrid Search (Dense + Sparse)"
          />
          
          <Box sx={{ minWidth: 200 }}>
            <Typography gutterBottom>Max Results: {maxResults}</Typography>
            <Slider
              value={maxResults}
              onChange={(_, value) => setMaxResults(value as number)}
              min={5}
              max={50}
              step={5}
              disabled={loading}
              marks
              valueLabelDisplay="auto"
            />
          </Box>
        </Box>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {searchTime > 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Found {results.length} results in {searchTime.toFixed(1)}ms
        </Typography>
      )}

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {results.map((result, index) => (
          <Card key={result.id} elevation={1}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'flex-start', mb: 1 }}>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                  {result.title || 'Untitled Document'}
                </Typography>
                <Chip
                  label={`Score: ${result.score.toFixed(3)}`}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
              </Box>
              
              {result.authors && result.authors.length > 0 && (
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Authors: {result.authors.join(', ')}
                </Typography>
              )}
              
              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                {result.year && (
                  <Chip label={result.year} size="small" variant="outlined" />
                )}
                {result.venue && (
                  <Chip label={result.venue} size="small" variant="outlined" />
                )}
                {result.page_number && (
                  <Chip label={`Page ${result.page_number}`} size="small" variant="outlined" />
                )}
              </Box>
              
              {result.abstract && (
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Abstract:</strong> {result.abstract.substring(0, 200)}
                  {result.abstract.length > 200 ? '...' : ''}
                </Typography>
              )}
              
              {result.text && (
                <Typography variant="body2" color="text.secondary">
                  {result.text.substring(0, 300)}
                  {result.text.length > 300 ? '...' : ''}
                </Typography>
              )}
            </CardContent>
          </Card>
        ))}
      </Box>

      {results.length === 0 && !loading && query && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h6" color="text.secondary">
            No results found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try a different query or search terms
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default SearchPage;