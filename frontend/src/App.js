import React, { useState, useEffect } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Tabs, 
  Tab, 
  Box,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import DataTable from './components/DataTable';
import VisualizationTab from './components/VisualizationTab';
import AddReasonTab from './components/AddReasonTab';
import UploadTab from './components/UploadTab';
import './App.css';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/data');
      const result = await response.json();
      setData(result.data || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="App">
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              JSW Energy - Deviation Dashboard
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 2 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="dashboard tabs">
              <Tab label="Data Table" />
              <Tab label="Visualizations" />
              <Tab label="Add Reason" />
              <Tab label="Upload Data" />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            <DataTable data={data} loading={loading} onDataUpdate={fetchData} />
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <VisualizationTab />
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <AddReasonTab />
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <UploadTab onUploadSuccess={fetchData} />
          </TabPanel>
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default App;