import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  Alert,
  CircularProgress,
  Divider
} from '@mui/material';

const AddReasonTab = () => {
  const [reasons, setReasons] = useState([]);
  const [newReason, setNewReason] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('info');

  useEffect(() => {
    fetchReasons();
  }, []);

  const fetchReasons = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/reasons');
      const result = await response.json();
      setReasons(result.reasons || []);
    } catch (error) {
      console.error('Error fetching reasons:', error);
      setMessage('Error fetching reasons');
      setMessageType('error');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!newReason.trim()) {
      setMessage('Please enter a reason');
      setMessageType('error');
      return;
    }

    if (reasons.includes(newReason.trim())) {
      setMessage('This reason already exists');
      setMessageType('error');
      return;
    }

    try {
      setSubmitting(true);
      const response = await fetch('/api/reasons', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ reason: newReason.trim() }),
      });

      if (response.ok) {
        setMessage('Reason added successfully!');
        setMessageType('success');
        setNewReason('');
        fetchReasons(); // Refresh the list
      } else {
        throw new Error('Failed to add reason');
      }
    } catch (error) {
      console.error('Error adding reason:', error);
      setMessage('Error adding reason');
      setMessageType('error');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Manage Reasons
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Add New Reason
        </Typography>
        
        {message && (
          <Alert 
            severity={messageType} 
            sx={{ mb: 2 }}
            onClose={() => setMessage('')}
          >
            {message}
          </Alert>
        )}

        <Box component="form" onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="New Reason"
            value={newReason}
            onChange={(e) => setNewReason(e.target.value)}
            placeholder="Enter a new reason for deviation"
            sx={{ mb: 2 }}
            disabled={submitting}
          />
          
          <Button
            type="submit"
            variant="contained"
            disabled={submitting || !newReason.trim()}
          >
            {submitting ? <CircularProgress size={24} /> : 'Add Reason'}
          </Button>
        </Box>
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Existing Reasons ({reasons.length})
        </Typography>
        
        <Divider sx={{ mb: 2 }} />
        
        {reasons.length > 0 ? (
          <List>
            {reasons.map((reason, index) => (
              <ListItem key={index} divider>
                <ListItemText primary={reason} />
              </ListItem>
            ))}
          </List>
        ) : (
          <Typography color="text.secondary">
            No reasons available
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default AddReasonTab;