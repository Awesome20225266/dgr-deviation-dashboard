#!/usr/bin/env python3
"""
FastAPI Backend for Deviation Dashboard
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import duckdb
import plotly.graph_objects as go
import plotly.express as px
import json
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase setup
from supabase import create_client, Client
from postgrest import APIError as PostgrestAPIError

SUPABASE_URL = "https://ubkcxehguactwwcarkae.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVia2N4ZWhndWFjdHd3Y2Fya2FlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMTU3OTYsImV4cCI6MjA2Nzc5MTc5Nn0.NPiJj_o-YervOE1dPxWRJhEI1fUwxT3Dptz-JszChLo"

def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize FastAPI app
app = FastAPI(title="Deviation Dashboard API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initial reason list
INITIAL_REASON_LIST = [
    "Rain / Storm",
    "Grid Failure",
    "Transmission Constraint",
    "Fuel Shortage",
    "Equipment Failure",
    "Maintenance",
    "Load Dispatch",
    "Others"
]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Deviation Dashboard API is running"}

@app.get("/api/data")
async def get_data():
    """Get all deviation data from DuckDB"""
    try:
        conn = duckdb.connect('dgr_data.duckdb')
        df = conn.execute("SELECT * FROM dgr_data").fetchdf()
        conn.close()
        
        # Convert DataFrame to JSON
        return {
            "data": df.to_dict('records'),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plants")
async def get_plants():
    """Get list of all plants"""
    try:
        conn = duckdb.connect('dgr_data.duckdb')
        plants = conn.execute("SELECT DISTINCT Plant FROM dgr_data ORDER BY Plant").fetchall()
        conn.close()
        
        return {"plants": [plant[0] for plant in plants]}
    except Exception as e:
        logger.error(f"Error fetching plants: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reasons")
async def get_reasons():
    """Get reasons from Supabase"""
    try:
        supabase = get_supabase_client()
        response = supabase.table('reasons').select('reason').execute()
        reasons = [item['reason'] for item in response.data]
        
        # Combine with initial reasons
        all_reasons = list(set(INITIAL_REASON_LIST + reasons))
        return {"reasons": sorted(all_reasons)}
    except Exception as e:
        logger.error(f"Error fetching reasons: {str(e)}")
        # Return initial reasons if Supabase fails
        return {"reasons": INITIAL_REASON_LIST}

@app.post("/api/reasons")
async def add_reason(reason_data: dict):
    """Add a new reason to Supabase"""
    try:
        reason = reason_data.get('reason', '').strip()
        if not reason:
            raise HTTPException(status_code=400, detail="Reason cannot be empty")
        
        supabase = get_supabase_client()
        supabase.table('reasons').insert({'reason': reason}).execute()
        
        return {"message": "Reason added successfully", "reason": reason}
    except Exception as e:
        logger.error(f"Error adding reason: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/filtered")
async def get_filtered_data(plant: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get filtered deviation data"""
    try:
        conn = duckdb.connect('dgr_data.duckdb')
        
        query = "SELECT * FROM dgr_data WHERE 1=1"
        params = []
        
        if plant and plant != "All":
            query += " AND Plant = ?"
            params.append(plant)
        
        if start_date:
            query += " AND Date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND Date <= ?"
            params.append(end_date)
        
        df = conn.execute(query, params).fetchdf()
        conn.close()
        
        return {
            "data": df.to_dict('records'),
            "columns": df.columns.tolist(),
            "count": len(df)
        }
    except Exception as e:
        logger.error(f"Error fetching filtered data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualizations/timeline")
async def get_timeline_chart(plant: Optional[str] = None):
    """Generate timeline visualization data"""
    try:
        conn = duckdb.connect('dgr_data.duckdb')
        
        query = "SELECT Date, Plant, SUM(\"Deviation (MWh)\") as total_deviation FROM dgr_data"
        params = []
        
        if plant and plant != "All":
            query += " WHERE Plant = ?"
            params.append(plant)
        
        query += " GROUP BY Date, Plant ORDER BY Date"
        
        df = conn.execute(query, params).fetchdf()
        conn.close()
        
        if df.empty:
            return {"data": [], "layout": {}}
        
        # Create Plotly figure
        fig = px.line(df, x='Date', y='total_deviation', color='Plant',
                     title='Deviation Timeline', 
                     labels={'total_deviation': 'Total Deviation (MWh)'})
        
        return fig.to_dict()
    except Exception as e:
        logger.error(f"Error generating timeline chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualizations/plant-summary")
async def get_plant_summary():
    """Generate plant summary visualization data"""
    try:
        conn = duckdb.connect('dgr_data.duckdb')
        
        df = conn.execute("""
            SELECT Plant, 
                   COUNT(*) as incident_count,
                   SUM("Deviation (MWh)") as total_deviation,
                   AVG("Deviation (MWh)") as avg_deviation
            FROM dgr_data 
            GROUP BY Plant 
            ORDER BY total_deviation DESC
        """).fetchdf()
        conn.close()
        
        if df.empty:
            return {"data": [], "layout": {}}
        
        # Create bar chart
        fig = px.bar(df, x='Plant', y='total_deviation',
                    title='Total Deviation by Plant',
                    labels={'total_deviation': 'Total Deviation (MWh)'})
        
        return fig.to_dict()
    except Exception as e:
        logger.error(f"Error generating plant summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/update")
async def update_data(update_data: dict):
    """Update deviation data"""
    try:
        # This would typically update the database
        # For now, just return success
        return {"message": "Data updated successfully"}
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process Excel file"""
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Process the uploaded data
        # This is a simplified version - you may need to adjust based on your data structure
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export")
async def export_data(format: str = "excel"):
    """Export data in specified format"""
    try:
        conn = duckdb.connect('dgr_data.duckdb')
        df = conn.execute("SELECT * FROM dgr_data").fetchdf()
        conn.close()
        
        if format.lower() == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Deviation Data')
            
            output.seek(0)
            return StreamingResponse(
                io.BytesIO(output.read()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=deviation_data.xlsx"}
            )
        else:
            # CSV export
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return StreamingResponse(
                io.StringIO(output.getvalue()),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=deviation_data.csv"}
            )
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)