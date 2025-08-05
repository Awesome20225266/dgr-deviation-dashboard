import React, { useState, useEffect } from 'react';
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Box,
  TextField,
  MenuItem,
  Button,
  Typography,
  CircularProgress,
  Grid
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs from 'dayjs';

const DataTable = ({ data, loading, onDataUpdate }) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [plants, setPlants] = useState([]);
  const [selectedPlant, setSelectedPlant] = useState('All');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [filteredData, setFilteredData] = useState([]);

  useEffect(() => {
    fetchPlants();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [data, selectedPlant, startDate, endDate]);

  const fetchPlants = async () => {
    try {
      const response = await fetch('/api/plants');
      const result = await response.json();
      setPlants(['All', ...result.plants]);
    } catch (error) {
      console.error('Error fetching plants:', error);
    }
  };

  const applyFilters = async () => {
    try {
      const params = new URLSearchParams();
      if (selectedPlant && selectedPlant !== 'All') {
        params.append('plant', selectedPlant);
      }
      if (startDate) {
        params.append('start_date', startDate.format('YYYY-MM-DD'));
      }
      if (endDate) {
        params.append('end_date', endDate.format('YYYY-MM-DD'));
      }

      const response = await fetch(`/api/data/filtered?${params}`);
      const result = await response.json();
      setFilteredData(result.data || []);
    } catch (error) {
      console.error('Error filtering data:', error);
      setFilteredData(data);
    }
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const exportData = async (format = 'excel') => {
    try {
      const response = await fetch(`/api/export?format=${format}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `deviation_data.${format === 'excel' ? 'xlsx' : 'csv'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting data:', error);
    }
  };

  const columns = filteredData.length > 0 ? Object.keys(filteredData[0]) : [];

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Deviation Data
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <TextField
              select
              fullWidth
              label="Plant"
              value={selectedPlant}
              onChange={(e) => setSelectedPlant(e.target.value)}
            >
              {plants.map((plant) => (
                <MenuItem key={plant} value={plant}>
                  {plant}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <DatePicker
              label="Start Date"
              value={startDate}
              onChange={setStartDate}
              renderInput={(params) => <TextField {...params} fullWidth />}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <DatePicker
              label="End Date"
              value={endDate}
              onChange={setEndDate}
              renderInput={(params) => <TextField {...params} fullWidth />}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Box display="flex" gap={1}>
              <Button
                variant="contained"
                onClick={() => exportData('excel')}
                size="small"
              >
                Export Excel
              </Button>
              <Button
                variant="outlined"
                onClick={() => exportData('csv')}
                size="small"
              >
                Export CSV
              </Button>
            </Box>
          </Grid>
        </Grid>

        <Typography variant="body2" color="text.secondary" gutterBottom>
          Total Records: {filteredData.length}
        </Typography>
      </Box>

      <Paper>
        <TableContainer>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                {columns.map((column) => (
                  <TableCell key={column} style={{ fontWeight: 'bold' }}>
                    {column}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredData
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((row, index) => (
                  <TableRow key={index} hover>
                    {columns.map((column) => (
                      <TableCell key={column}>
                        {row[column] !== null && row[column] !== undefined 
                          ? String(row[column]) 
                          : ''}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <TablePagination
          rowsPerPageOptions={[10, 25, 50, 100]}
          component="div"
          count={filteredData.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
    </LocalizationProvider>
  );
};

export default DataTable;