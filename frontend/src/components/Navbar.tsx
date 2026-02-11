import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Search, QuestionAnswer, Upload, Analytics } from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    { label: 'Search', path: '/search', icon: <Search /> },
    { label: 'Q&A', path: '/qa', icon: <QuestionAnswer /> },
    { label: 'Upload', path: '/upload', icon: <Upload /> },
    { label: 'Analytics', path: '/analytics', icon: <Analytics /> },
  ];

  return (
    <AppBar position="fixed" elevation={2}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Scientific RAG System
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          {menuItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              startIcon={item.icon}
              onClick={() => navigate(item.path)}
              sx={{
                backgroundColor: location.pathname === item.path ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                borderRadius: 1,
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;