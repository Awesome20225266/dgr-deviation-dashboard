import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  TextField,
  MenuItem,
  CircularProgress
} from '@mui/material';
import Plot from 'react-plotly.js';

const VisualizationTab = () => {
  const [plants, setPlants] = useState([]);
  const [selectedPlant, setSelectedPlant] = useState('All');
  const [timelineData, setTimelineData] = useState(null);
  const [plantSummaryData, setPlantSummaryData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPlants();
    fetchVisualizationData();
  }, []);

  useEffect(() => {
    fetchVisualizationData();
  }, [selectedPlant]);

  const fetchPlants = async () => {
    try {
      const response = await fetch('/api/plants');
      const result = await response.json();
      setPlants(['All', ...result.plants]);
    } catch (error) {
      console.error('Error fetching plants:', error);
    }
  };

  const fetchVisualizationData = async () => {
    try {
      setLoading(true);
      
      // Fetch timeline data
      const timelineParams = selectedPlant !== 'All' ? `?plant=${selectedPlant}` : '';
      const timelineResponse = await fetch(`/api/visualizations/timeline${timelineParams}`);
      const timelineResult = await timelineResponse.json();
      setTimelineData(timelineResult);

      // Fetch plant summary data
      const summaryResponse = await fetch('/api/visualizations/plant-summary');
      const summaryResult = await summaryResponse.json();
      setPlantSummaryData(summaryResult);
      
    } catch (error) {
      console.error('Error fetching visualization data:', error);
    } finally {
      setLoading(false);
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
        Visualizations
      </Typography>

      <Box sx={{ mb: 3 }}>
        <TextField
          select
          label="Filter by Plant"
          value={selectedPlant}
          onChange={(e) => setSelectedPlant(e.target.value)}
          sx={{ minWidth: 200 }}
        >
          {plants.map((plant) => (
            <MenuItem key={plant} value={plant}>
              {plant}
            </MenuItem>
          ))}
        </TextField>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Deviation Timeline
            </Typography>
            {timelineData && timelineData.data ? (
              <Plot
                data={timelineData.data}
                layout={{
                  ...timelineData.layout,
                  autosize: true,
                  height: 400
                }}
                useResizeHandler
                style={{ width: '100%' }}
              />
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="400px">
                <Typography color="text.secondary">No timeline data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Plant Summary
            </Typography>
            {plantSummaryData && plantSummaryData.data ? (
              <Plot
                data={plantSummaryData.data}
                layout={{
                  ...plantSummaryData.layout,
                  autosize: true,
                  height: 400
                }}
                useResizeHandler
                style={{ width: '100%' }}
              />
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="400px">
                <Typography color="text.secondary">No plant summary data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default VisualizationTab;