import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Alert,
  CircularProgress,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const UploadTab = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('info');
  const [uploadResult, setUploadResult] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = [
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-excel',
      'text/csv'
    ];
    
    if (!allowedTypes.includes(file.type)) {
      setMessage('Please upload an Excel file (.xlsx, .xls) or CSV file');
      setMessageType('error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploading(true);
      setUploadProgress(0);
      setMessage('');
      setUploadResult(null);

      // Simulate progress (since we can't track real progress easily with fetch)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.ok) {
        const result = await response.json();
        setUploadResult(result);
        setMessage('File uploaded successfully!');
        setMessageType('success');
        
        // Call the success callback
        if (onUploadSuccess) {
          onUploadSuccess();
        }
      } else {
        const errorResult = await response.json();
        throw new Error(errorResult.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage(`Upload failed: ${error.message}`);
      setMessageType('error');
      setUploadProgress(0);
    } finally {
      setUploading(false);
      // Reset file input
      event.target.value = '';
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Upload Data
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Upload Excel or CSV File
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Upload your deviation data file. Supported formats: Excel (.xlsx, .xls) and CSV (.csv)
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

        <Box sx={{ mb: 3 }}>
          <Button
            component="label"
            variant="contained"
            startIcon={<CloudUploadIcon />}
            disabled={uploading}
            size="large"
          >
            {uploading ? 'Uploading...' : 'Choose File'}
            <VisuallyHiddenInput
              type="file"
              accept=".xlsx,.xls,.csv"
              onChange={handleFileUpload}
            />
          </Button>
        </Box>

        {uploading && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Upload Progress: {uploadProgress}%
            </Typography>
            <LinearProgress variant="determinate" value={uploadProgress} />
          </Box>
        )}

        {uploadResult && (
          <Paper variant="outlined" sx={{ p: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
            <Typography variant="h6" gutterBottom>
              Upload Summary
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="Filename" 
                  secondary={uploadResult.filename} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Rows Processed" 
                  secondary={uploadResult.rows} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Columns" 
                  secondary={uploadResult.columns?.join(', ') || 'N/A'} 
                />
              </ListItem>
            </List>
          </Paper>
        )}
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          File Format Guidelines
        </Typography>
        
        <Divider sx={{ mb: 2 }} />
        
        <Typography variant="body1" paragraph>
          <strong>Required Columns:</strong>
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText primary="• Date - Date of the deviation" />
          </ListItem>
          <ListItem>
            <ListItemText primary="• Plant - Name of the power plant" />
          </ListItem>
          <ListItem>
            <ListItemText primary="• Deviation (MWh) - Amount of deviation in MWh" />
          </ListItem>
          <ListItem>
            <ListItemText primary="• Reason - Reason for the deviation (optional)" />
          </ListItem>
        </List>
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          Note: The uploaded data will be processed and added to the existing database. 
          Please ensure your data follows the expected format to avoid import errors.
        </Typography>
      </Paper>
    </Box>
  );
};

export default UploadTab;