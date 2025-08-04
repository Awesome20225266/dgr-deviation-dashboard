# main.py (complete updated script)

import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
import base64
import matplotlib
import numpy as np
from datetime import datetime
# --- SUPABASE CLIENT SETUP ---
from supabase import create_client, Client
from postgrest import APIError as PostgrestAPIError

SUPABASE_URL = "https://ubkcxehguactwwcarkae.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVia2N4ZWhndWFjdHd3Y2Fya2FlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMTU3OTYsImV4cCI6MjA2Nzc5MTc5Nn0.NPiJj_o-YervOE1dPxWRJhEI1fUwxT3Dptz-JszChLo"

def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

from visualisation_tab import render_visualisation_tab, get_reasons, INITIAL_REASON_LIST
import io
import pandas as pd
import traceback
import datetime
import openpyxl
import re


def to_excel_bytes(df, reason_options):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Deviation Data')
        workbook = writer.book
        worksheet = writer.sheets['Deviation Data']
        
        # Write reason_options to a hidden column after the data
        last_col = len(df.columns)  # 0-based index, next column index
        reason_col_letter = chr(65 + last_col)  # e.g., if 3 cols (A-C), next is D
        for i, reason in enumerate(reason_options, start=1):
            worksheet.write(f'{reason_col_letter}{i}', reason)
        
        # Hide the reason column
        worksheet.set_column(last_col, last_col, None, None, {'hidden': True})
        
        # Find the "Reason" column index
        reason_col = df.columns.get_loc("Reason")
        
        # Apply dropdown to all rows in Reason column (first_row=1 for data rows)
        worksheet.data_validation(
            first_row=1, first_col=reason_col,
            last_row=len(df), last_col=reason_col,
            options={
                'validate': 'list',
                'source': f'=${reason_col_letter}$1:${reason_col_letter}${len(reason_options)}',
                'input_message': 'Select a reason',
                'error_message': 'Choose only from list'
            }
        )
    output.seek(0)
    return output.getvalue()

def safe_db_execute(con, query, params):
    try:
        return con.execute(query, params).df()
    except Exception as e:
        st.error(f"Database error: {e}")
        traceback.print_exc()
        return pd.DataFrame()

DB_PATH = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
MAPPING_SHEET = "Mapping Sheet.xlsx"

st.set_page_config(page_title="DGR Deviation Dashboard", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = "JSW-Energy-completes-acquisition-of-4.7-GW-renewable-energy-platform-from-O2-Power.jpg"
img_base64 = get_base64_of_bin_file(img_path)
background_css = f"""
<style>
[data-testid="stAppViewContainer"] > .main::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 0;
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: 500px auto;
    background-repeat: no-repeat;
    background-position: center 35%;
    opacity: 0.13;
    pointer-events: none;
}}
[data-testid="stHeader"] {{
    background: rgba(255,255,255,0.85);
}}
.block-container {{
    background: rgba(255,255,255,0.94) !important;
    border-radius: 18px;
    padding: 8px 22px;
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Global Custom CSS for Modern UI
custom_css = """
<style>
    /* Font and Color Palette */
    body, .stApp {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #333333;
        background-color: #f9f9fb;
    }

    /* Section Headers */
    .section-header {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #007bff;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
    }
    button[kind="primary"]:hover {
        background-color: #0056b3;
    }
    button[kind="secondary"] {
        background-color: #6c757d;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
    }
    button[kind="secondary"]:hover {
        background-color: #5a6268;
    }

    /* AGGrid Styling */
    .ag-header-cell-label {
        font-weight: bold;
        color: #007bff;
    }
    .ag-row-even {
        background-color: #f8f9fa;
    }
    .ag-row-odd {
        background-color: #ffffff;
    }
    .ag-row-hover {
        background-color: #e2e6ea !important;
    }
    .ag-cell {
        line-height: 1.5 !important;
    }

    /* Cards/Containers */
    .st-expander, .st-container {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        padding: 10px;
        margin-bottom: 15px;
    }

    /* Messages */
    .st-info, .st-warning, .st-error {
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 15px;
    }

    /* Spacing */
    .divider {
        margin: 20px 0;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Define deviation_to_color function
import matplotlib

def deviation_to_color(val):
    try:
        v = float(val.replace('%', '') if isinstance(val, str) else val)
    except:
        return ""
    max_neg = -100  # Adjust to your worst-case negative deviation
    norm = min(abs(v) / abs(max_neg), 1)
    rgba = matplotlib.cm.Reds(norm)
    r, g, b, _ = [int(x * 255) for x in rgba]
    # If too dark, force text to white, else black
    text_color = "white" if norm > 0.7 else "black"
    return f"background-color: rgb({r},{g},{b}); color: {text_color};"

st.title("📊 DGR Deviation Dashboard")
st.markdown("Analyze inverter/block-wise deviation across plants using data from DuckDB.")

@st.cache_data
def load_mapping():
    return pd.read_excel(MAPPING_SHEET)

mapping_df = load_mapping()
plants_available = mapping_df["Plant_Name"].unique().tolist()

# --- Default Plant/Date Selection: No pre-selection [Requirement 1]
# Use session state to persist select_all
if "select_all" not in st.session_state:
    st.session_state.select_all = False

st.session_state.select_all = st.checkbox("Select All Plants", value=st.session_state.select_all)

if st.session_state.select_all:
    default_plants = plants_available
else:
    default_plants = []

plant_select = st.multiselect("Select Plant(s)", options=plants_available, default=default_plants)

if not plant_select:
    st.warning("Please select at least one plant.")
    # Allow date selection, but check later

# --- Threshold ---
if "threshold" not in st.session_state:
    st.session_state["threshold"] = -3.0
threshold = st.number_input("Deviation Threshold (e.g., -3 means ≤ -3%)",
                            value=st.session_state["threshold"], step=0.1, key="threshold")

# --- Date Range: No default selection [Requirement 1]
with duckdb.connect(DB_PATH) as con:
    date_query = f"""
        SELECT DISTINCT date FROM {TABLE_NAME}
        ORDER BY date
    """
    dates_result = con.execute(date_query).fetchall()
    all_dates = [d[0] for d in dates_result]

if all_dates:
    date_min = min(all_dates)
    date_max = max(all_dates)
    date_start = st.date_input("Start Date", value=None, min_value=date_min, max_value=date_max)
    date_end = st.date_input("End Date", value=None, min_value=date_min, max_value=date_max)
else:
    st.warning("No data found in the database.")
    st.stop()

if date_start and date_end and date_start > date_end:
    st.warning("Start date must be before end date.")
    st.stop()

# --- Call this after all widgets ---
if not plant_select or not date_start or not date_end:
    st.warning("Please select Plant(s) and Date range to view dashboard data.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([ 
    "Generate Table", "Generate Ranking", "Visualisation", "Portfolio Deep Analytics", "Visual Summary", "Comment Map", "Add Reason"
])
# --- TAB 1: GENERATE TABLE ---
with tab1:
    st.markdown('<div class="section-header"><h3>📋 Generate Table</h3></div>', unsafe_allow_html=True)
    if st.button("Generate Table", type="primary"):
        # [Requirement 1] Check for plant and date selection
        if not plant_select or not date_start or not date_end:
            st.warning("Please select at least one plant and a date range to continue.")
        else:
            with st.spinner("Processing..."):
                with duckdb.connect(DB_PATH) as con:
                    query = f"""
                        SELECT plant, date, input_name, value
                        FROM {TABLE_NAME}
                        WHERE plant IN ({','.join(['?'] * len(plant_select))})
                          AND date BETWEEN ? AND ?
                    """
                    params = plant_select + [date_start, date_end]
                    df = safe_db_execute(con, query, params)

                if df.empty:
                    st.warning("No data found in selected range.")
                else:
                    threshold_label = f"No. of days ≤ {threshold}%"

                    if len(plant_select) == 1:
                        plant_name = plant_select[0]
                        mapping_row = mapping_df[mapping_df["Plant_Name"] == plant_name].iloc[0]
                        start_col = mapping_row["Data_Start_Col"]
                        end_col = mapping_row["Data_End_Col"]
                        equipment_names = df["input_name"].unique().tolist()
                        if start_col in equipment_names and end_col in equipment_names:
                            sidx = equipment_names.index(start_col)
                            eidx = equipment_names.index(end_col)
                            mapped_equipment = equipment_names[sidx:eidx+1]
                        else:
                            mapped_equipment = equipment_names

                        table_rows = []
                        serial = 1
                        if date_start == date_end:
                            day = pd.to_datetime(date_start)
                            for eq in mapped_equipment:
                                eq_val = df[(df["input_name"] == eq) & (df["date"] == day)]
                                value = eq_val["value"].iloc[0] if not eq_val.empty else None
                                count = 1 if (not eq_val.empty and value <= threshold) else 0
                                if value is not None:  # [Requirement 6] Include -100% deviation
                                    table_rows.append([
                                        serial, plant_name, eq, f"{value:.2f}%", 1, count
                                    ])
                                    serial += 1
                        else:
                            for eq in mapped_equipment:
                                eq_df = df[df["input_name"] == eq]
                                if not eq_df.empty:
                                    avg_deviation = eq_df["value"].mean()
                                    num_days = eq_df["date"].nunique()
                                    count = (eq_df["value"] <= threshold).sum()
                                    table_rows.append([
                                        serial, plant_name, eq, f"{avg_deviation:.2f}%", num_days, count
                                    ])
                                    serial += 1
                        outdf = pd.DataFrame(table_rows, columns=[
                            "Serial No.", "Plant_Name", "Equipment_Name", "Avg %Deviation", "No. of Days in Range", threshold_label
                        ])
                        if outdf.empty:
                            st.info("No underperformers found for the selected criteria.")
                        else:
                            st.markdown("### Equipment-wise Deviation Table")
                            # Format Avg %Deviation to 3 decimals (strip % for formatting, then add back)
                            outdf["Avg %Deviation"] = outdf["Avg %Deviation"].str.replace('%', '').astype(float).apply(lambda x: f"{x:.3f}%")
                            # Row coloring by Avg %Deviation
                            styled_outdf = outdf.style.applymap(deviation_to_color, subset=["Avg %Deviation"])
                            st.dataframe(styled_outdf, use_container_width=True)
                    else:
                        rows = []
                        serial = 1
                        for plant in df['plant'].unique():
                            plant_df = df[df['plant'] == plant]
                            total_input_cells = plant_df['value'].notnull().sum()
                            inputs_deviated = (plant_df["value"] <= threshold).sum()
                            norm_deviation = 100 * inputs_deviated / total_input_cells if total_input_cells > 0 else 0
                            avg_deviation = plant_df[plant_df["value"] <= threshold]["value"].mean() if inputs_deviated > 0 else 0
                            rows.append([
                                serial,
                                plant,
                                f"{avg_deviation:.2f}%",
                                inputs_deviated,
                                total_input_cells,
                                f"{norm_deviation:.2f}%",
                                inputs_deviated
                            ])
                            serial += 1
                        outdf = pd.DataFrame(rows, columns=[
                            "Serial No.", "Plant_Name", "%Deviation", "No. of Inputs Deviated", "Total Inputs", "Normalised Deviation (%)", f"No. of Inputs ≤ {threshold}%"
                        ])
                        if outdf.empty:
                            st.info("No deviations below threshold found for any plant in selected range.")
                        else:
                            st.markdown("### Plant-wise Deviation Summary")
                            # Format %Deviation to 3 decimals
                            outdf["%Deviation"] = outdf["%Deviation"].str.replace('%', '').astype(float).apply(lambda x: f"{x:.3f}%")
                            # Row coloring
                            styled_outdf = outdf.style.applymap(deviation_to_color, subset=["%Deviation"])
                            st.dataframe(styled_outdf, use_container_width=True)

# --- TAB 2: GENERATE RANKING ---
with tab2:
    st.markdown('<div class="section-header"><h3>🏆 Generate Ranking</h3></div>', unsafe_allow_html=True)
    if st.button("Generate Ranking", type="primary"):
        # [Requirement 1] Check for plant and date selection
        if not plant_select or not date_start or not date_end:
            st.warning("Please select at least one plant and a date range to continue.")
        else:
            with st.spinner("Ranking..."):
                with duckdb.connect(DB_PATH) as con:
                    query = f"""
                        SELECT plant, date, input_name, value
                        FROM {TABLE_NAME}
                        WHERE plant IN ({','.join(['?'] * len(plant_select))})
                          AND date BETWEEN ? AND ?
                    """
                    params = plant_select + [date_start, date_end]
                    df = safe_db_execute(con, query, params)

                if df.empty:
                    st.warning("No data found in selected range.")
                else:
                    if len(plant_select) == 1:
                        plant_name = plant_select[0]
                        equipment_names = df["input_name"].unique().tolist()
                        table_rows = []
                        threshold_label = f"No. of days ≤ {threshold}%"
                        if date_start == date_end:
                            day = pd.to_datetime(date_start)
                            for eq in equipment_names:
                                eq_vals = df[(df["input_name"] == eq) & (df["date"] == day)]
                                value = eq_vals["value"].iloc[0] if not eq_vals.empty else None
                                count = 1 if (not eq_vals.empty and value <= threshold) else 0
                                if value is not None:  # [Requirement 6] Include -100% deviation
                                    table_rows.append([eq, value if value is not None else 0, 1, count])
                        else:
                            for eq in equipment_names:
                                eq_df = df[df["input_name"] == eq]
                                avg_dev = eq_df["value"].mean() if not eq_df.empty else 0
                                n_days = eq_df["date"].nunique()
                                count = (eq_df["value"] <= threshold).sum() if not eq_df.empty else 0
                                table_rows.append([eq, avg_dev, n_days, count])
                        outdf = pd.DataFrame(table_rows, columns=["Equipment Name", "Deviation", "No. of Days", threshold_label])

                        outdf = outdf.sort_values("Deviation", ascending=True).reset_index(drop=True)
                        outdf["Rank"] = outdf.index + 1
                        outdf["Deviation"] = outdf["Deviation"].map(lambda x: f"{x:.3f}%" if pd.notnull(x) else "-")
                        outdf[threshold_label] = outdf[threshold_label].astype(int).fillna(0)

                        st.markdown("### 🌟 Equipment-wise Deviation Ranking")
                        # Row coloring
                        styled_outdf = outdf.style.applymap(deviation_to_color, subset=["Deviation"])
                        st.dataframe(styled_outdf, use_container_width=True)

                        dev_numeric = [float(str(x).replace('%','')) for x in outdf["Deviation"] if x not in ["-", None]]
                        equipment_names = outdf["Equipment Name"].tolist()

                        bar_colors = []
                        if dev_numeric:
                            min_dev = min(dev_numeric)
                            max_dev = max(dev_numeric)
                            if max_dev == min_dev:
                                norm_values = [0.5] * len(dev_numeric)
                            else:
                                norm_values = [(v - min_dev) / (max_dev - min_dev) for v in dev_numeric]
                            cmap = matplotlib.colormaps['RdYlGn']
                            bar_colors = [matplotlib.colors.rgb2hex(cmap(norm)) for norm in norm_values]
                        else:
                            bar_colors = None

                        fig = go.Figure(go.Bar(
                            y=equipment_names,
                            x=dev_numeric,
                            orientation='h',
                            marker_color=bar_colors if bar_colors else 'grey',
                            text=[f"{x:.2f}%" for x in dev_numeric],
                            textposition='outside',
                            insidetextanchor='end',
                            hovertemplate="Equipment: %{y}<br>Deviation: %{x:.2f}%<extra></extra>"
                        ))
                        fig.update_yaxes(range=[-0.5, len(outdf)-0.5], autorange="reversed")
                        fig.update_layout(
                            title="🏆 Equipment Ranking by Deviation (Red = Worst, Green = Best)",
                            xaxis_title="Deviation (%)",
                            yaxis_title="Equipment",
                            font=dict(family="Segoe UI, Arial", size=16),
                            xaxis=dict(tickformat=".2f", autorange=True),
                            yaxis=dict(autorange=True),
                            plot_bgcolor='white',
                            margin=dict(t=70, l=150, r=40),
                            height=100 + 60 * len(outdf)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        rows = []
                        for plant in df['plant'].unique():
                            plant_df = df[df['plant'] == plant]
                            total_input_cells = plant_df['value'].notnull().sum()
                            inputs_deviated = (plant_df["value"] <= threshold).sum()
                            norm_deviation = 100 * inputs_deviated / total_input_cells if total_input_cells > 0 else 0
                            rows.append([plant, norm_deviation, inputs_deviated])
                        ranked = pd.DataFrame(rows, columns=["Plant", "Normalised Deviation (%)", f"No. of Inputs ≤ {threshold}%"])

                        ranked["AbsDeviation"] = ranked["Normalised Deviation (%)"].abs()
                        ranked = ranked.sort_values("AbsDeviation").reset_index(drop=True)
                        ranked['Rank'] = ranked.index + 1

                        st.markdown("### 🌟 Plant Normalised Deviation Ranking (Lower ABS = Better, Higher ABS = Redder)")
                        # Format to 3 decimals
                        ranked["Normalised Deviation (%)"] = ranked["Normalised Deviation (%)"].apply(lambda x: f"{x:.3f}%")
                        # Row coloring on Normalised Deviation (%)
                        styled_ranked = ranked.drop(columns="AbsDeviation").style.applymap(deviation_to_color, subset=["Normalised Deviation (%)"])
                        st.dataframe(styled_ranked, use_container_width=True)

                        abs_devs = ranked["AbsDeviation"]
                        dev_numeric = ranked["Normalised Deviation (%)"].str.replace('%', '').astype(float).tolist()
                        if len(abs_devs) > 0:
                            cmap = matplotlib.colormaps['RdYlGn_r']
                            min_dev = np.min(abs_devs)
                            max_dev = np.max(abs_devs)
                            if min_dev == max_dev:
                                norm = np.ones(len(abs_devs)) * 0.5
                            else:
                                norm = (abs_devs - min_dev) / (max_dev - min_dev)
                            bar_colors = [matplotlib.colors.rgb2hex(cmap(x)) for x in norm]
                        else:
                            bar_colors = None
                        text_labels = [f"Rank {row.Rank}, {row['Normalised Deviation (%)']}" for idx, row in ranked.iterrows()]
                        fig = go.Figure(go.Bar(
                            y=ranked["Plant"],
                            x=dev_numeric,
                            orientation='h',
                            marker_color=bar_colors if bar_colors else 'grey',
                            text=text_labels,
                            textposition='outside',
                            insidetextanchor='end',
                            hovertemplate="Plant: %{y}<br>Deviation: %{x:.2f}%<extra></extra>"
                        ))
                        fig.update_yaxes(range=[-0.5, len(ranked)-0.5], autorange="reversed")
                        fig.update_layout(
                            title="🏆 Plant Ranking by |Normalised Deviation| (Lower ABS = Greener, Higher ABS = Redder)",
                            xaxis_title="Normalised Deviation (%)",
                            yaxis_title="Plant",
                            font=dict(family="Segoe UI, Arial", size=16),
                            xaxis=dict(tickformat=".2f", autorange=True),
                            yaxis=dict(autorange=True),
                            plot_bgcolor='white',
                            margin=dict(t=70, l=150, r=40),
                            height=100 + 60 * len(ranked)
                        )
                        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: VISUALISATION ---
with tab3:
    st.markdown('<div class="section-header"><h3>📊 Visualisation</h3></div>', unsafe_allow_html=True)
    # [Requirement 1] Check for plant and date selection
    if not plant_select or not date_start or not date_end:
        st.warning("Please select at least one plant and a date range to continue.")
    else:
        render_visualisation_tab(plant_select, date_start, date_end, threshold)

# --- TAB 4: PORTFOLIO DEEP ANALYTICS ---
from visualisation_tab import REASON_COLOR, get_reasons, INITIAL_REASON_LIST

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from collections import defaultdict  # Added import for defaultdict
import time

with tab4:
    st.markdown('<div class="section-header"><h3>📈 Portfolio Deep Analytics</h3></div>', unsafe_allow_html=True)
    st.markdown("Uniform Calculation for Portfolio Insights")
    st.divider()

    if "portfolio_submit_success" not in st.session_state:
        st.session_state["portfolio_submit_success"] = False
    if st.session_state["portfolio_submit_success"]:
        st.session_state["portfolio_tag_eq"] = []
        st.session_state["portfolio_reason"] = None
        st.session_state["portfolio_tag_daterange"] = ()
        st.session_state["portfolio_tag_comment"] = ""
        st.session_state["portfolio_submit_success"] = False

    # Use main selectors
    portfolio_plants = plant_select
    portfolio_threshold = threshold
    portfolio_date_start = date_start
    portfolio_date_end = date_end

    # [Requirement 1] Check for plant and date selection
    if not portfolio_plants or not portfolio_date_start or not portfolio_date_end:
        st.warning("Please select at least one plant and a date range to continue.")
    else:
        with st.spinner("Fetching data..."):
            with duckdb.connect(DB_PATH) as con:
                query = f"""
                    SELECT plant, date, input_name, value
                    FROM {TABLE_NAME}
                    WHERE plant IN ({','.join(['?'] * len(portfolio_plants))})
                      AND date BETWEEN ? AND ?
                    """
                params = portfolio_plants + [portfolio_date_start, portfolio_date_end]
                df_portfolio = safe_db_execute(con, query, params)

        if df_portfolio.empty:
            st.warning("No data found for selected criteria.")
        else:
            # --- Calculate average deviation per Plant-Equipment (ALL values, not filtered) ---
            avg_dev_df = (
                df_portfolio.groupby(['plant', 'input_name'])['value']
                .mean()
                .reset_index(name='Avg Deviation (%)')
            )

            # --- Filter to show only equipment where average deviation < threshold ---
            avg_dev_df = avg_dev_df[avg_dev_df['Avg Deviation (%)'] < portfolio_threshold]

            if avg_dev_df.empty:
                st.info(f"No equipment found with Avg Deviation < {portfolio_threshold}% in this range.")
            else:
                # --- Bar chart coloring and layout ---
                import plotly.express as px

                plant_list = avg_dev_df['plant'].unique().tolist()
                px_palette = px.colors.qualitative.Plotly
                plant_color_map = {plant: px_palette[i % len(px_palette)] for i, plant in enumerate(plant_list)}

                # Assign color per bar by plant
                bar_colors = avg_dev_df['plant'].map(plant_color_map)
                if len(portfolio_plants) == 1:
                    dev_values = avg_dev_df['Avg Deviation (%)'].tolist()
                    min_dev = min(dev_values)
                    max_dev = max(dev_values)
                    if max_dev == min_dev:
                        norm_values = [0.5] * len(dev_values)
                    else:
                        norm_values = [(v - min_dev) / (max_dev - min_dev) for v in dev_values]
                    cmap = matplotlib.colormaps['RdYlGn']
                    bar_colors = [matplotlib.colors.rgb2hex(cmap(norm)) for norm in norm_values]

                fig = go.Figure(
                    go.Bar(
                        x=avg_dev_df['input_name'],
                        y=avg_dev_df['Avg Deviation (%)'],
                        marker_color=bar_colors,
                        text=[f"{v:.3f}%" for v in avg_dev_df['Avg Deviation (%)']],
                        textposition='auto',
                        hovertemplate="<b>%{x}</b><br>Deviation: %{y:.3f}%<extra></extra>",
                    )
                )

                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.add_hline(y=portfolio_threshold, line_dash="dot", line_color="red")

                fig.update_layout(
                    title="Equipment Deviation by Plant (Avg of All Days, Only < Threshold)",
                    xaxis_title="Equipment",
                    yaxis_title="Avg Deviation (%)",
                    plot_bgcolor='white',
                    height=600,
                    xaxis_tickangle=-40,
                    bargap=0.18,
                    showlegend=False,
                    xaxis=dict(autorange=True),
                    yaxis=dict(autorange=True)
                )

                # Visually group by plant (optional: shaded background rectangles)
                curr_x = 0
                for plant in plant_list:
                    count = (avg_dev_df['plant'] == plant).sum()
                    fig.add_vrect(
                        x0=curr_x-0.5,
                        x1=curr_x+count-1+0.5,
                        fillcolor=plant_color_map[plant],
                        opacity=0.08,
                        line_width=0
                    )
                    curr_x += count

                # Add custom legend
                legend_items = [
                    f"<span style='color:{plant_color_map[plant]}; font-weight:bold;'>⬤</span> {plant} ({(avg_dev_df['plant'] == plant).sum()})"
                    for plant in plant_list
                ]
                legend_html = "<br>".join(legend_items)
                fig.update_layout(
                    margin=dict(t=70, l=60, r=20, b=120),
                    annotations=[
                        dict(
                            text=legend_html,
                            align='left',
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.98, y=0.98,  # INSIDE the plot, near top-right
                            xanchor='right', yanchor='top',
                            bordercolor="#ccc",
                            borderwidth=1,
                            bgcolor="white",
                            opacity=0.93,
                            font=dict(size=13)
                        )
                    ]
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Deviation", f"{avg_dev_df['Avg Deviation (%)'].mean():.3f}%", delta=f"{(avg_dev_df['Avg Deviation (%)'].mean() - threshold):.3f}% from threshold", delta_color="inverse")
                with col2:
                    st.metric("Worst Deviation", f"{avg_dev_df['Avg Deviation (%)'].min():.3f}%", delta=f"{(avg_dev_df['Avg Deviation (%)'].min() - threshold):.3f}% from threshold", delta_color="inverse")
                with col3:
                    st.metric("Number of Equipments", len(avg_dev_df))

                st.plotly_chart(fig, use_container_width=True)

                # --- Download Button (same uniform logic) ---
                download_df = avg_dev_df.rename(columns={"plant": "Plant Name", "input_name": "Equipment Name"})
                download_df["Avg Deviation (%)"] = download_df["Avg Deviation (%)"].round(3)
                # Add empty columns for upload
                download_df["Fault Start Date"] = ""
                download_df["Fault End Date"] = ""
                download_df["Reason"] = ""
                download_df["Custom Reason"] = ""
                download_df["Comment"] = ""

                REASON_LIST = get_reasons()

                st.download_button(
                label="Download Deviation Data (Excel)",
                data=to_excel_bytes(download_df, REASON_LIST),
                file_name=f"Portfolio_Avg_Deviations_{portfolio_date_start}_{portfolio_date_end}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Bulk Reason Upload
                import io
                import pandas as pd
                from datetime import datetime

                with st.expander("📥 Bulk Reason Upload (Excel/CSV)", expanded=False):
                    st.markdown("""
                **How this works:**

                1. Click **"Download Deviation Data (Excel)"** above – this gives you underperforming equipment with deviation prefilled.
                2. For equipment you want to tag, fill in:
                    - **Fault Start Date** (YYYY-MM-DD, required)
                    - **Fault End Date** (YYYY-MM-DD, optional for date ranges)
                    - **Reason** (choose from list, or use "Others")
                    - **Custom Reason** (only if "Others")
                    - **Comment** (required)
                3. You can fill for any/all rows. Leave others blank.
                4. **Upload the same file** (xls, xlsx, csv) below.  
                    _Only rows with Reason and Comment will be processed. Timestamp is automatic if not present._

                    **Note:** For date range, one record per range will be tagged in DB. If only Fault Start Date is filled, only that date will be tagged.
                """)
                if "upload_key" not in st.session_state:
                    st.session_state.upload_key = 0

                uploaded_file = st.file_uploader(
                    "Upload filled deviation file (xls, xlsx, csv)", 
                    type=['xls', 'xlsx', 'csv'], 
                    key=f"bulk_upload_deviation_{st.session_state.upload_key}"
                )

                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df_upload = pd.read_csv(uploaded_file)
                        else:
                            uploaded_file.seek(0)
                            wb = openpyxl.load_workbook(uploaded_file, data_only=True)
                            ws = wb.active
                            data = [[cell.value for cell in row] for row in ws.rows if any(cell.value for cell in row)]
                            cols = data[0]
                            data_rows = data[1:]
                            df_upload = pd.DataFrame(data_rows, columns=cols)

                        # Clean columns and check
                        required_cols = [
                            "Plant Name", "Equipment Name", "Avg Deviation (%)",
                            "Fault Start Date", "Fault End Date", "Reason", "Custom Reason", "Comment"
                        ]
                        missing_cols = [c for c in required_cols if c not in df_upload.columns]
                        if missing_cols:
                            st.error(f"Missing columns in upload: {missing_cols}")
                        else:
                            valid_rows = []
                            skipped = []
                            row_errors = []
                            for idx, row in df_upload.iterrows():
                                plant = str(row.get("Plant Name", "")).strip()
                                equip = str(row.get("Equipment Name", "")).strip()
                                reason = str(row.get("Reason", "")).strip().lower()  # Case-insensitive
                                custom_reason = str(row.get("Custom Reason", "")).strip()
                                comment = str(row.get("Comment", "")).strip()
                                start_date_str = str(row.get("Fault Start Date", "")).strip()
                                end_date_str = str(row.get("Fault End Date", "")).strip()

                                # Handle nan/NaT as empty
                                if pd.isna(reason) or reason == 'nan':
                                    reason = ''
                                if pd.isna(custom_reason) or custom_reason == 'nan':
                                    custom_reason = ''
                                if pd.isna(comment) or comment == 'nan':
                                    comment = ''
                                if pd.isna(start_date_str) or start_date_str == 'nan':
                                    start_date_str = ''
                                if pd.isna(end_date_str) or end_date_str == 'nan':
                                    end_date_str = ''

                                # Skip if missing required fields
                                if not plant or not equip or not reason or not comment or not start_date_str:
                                    skipped.append(idx + 2)
                                    continue

                                # Robust date parsing
                                def parse_date(date_str):
                                    if not date_str:
                                        return None
                                    try:
                                        if isinstance(date_str, datetime):
                                            return date_str.strftime('%Y-%m-%d')
                                        try:
                                            serial = float(date_str)
                                            dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
                                            return dt.strftime('%Y-%m-%d')
                                        except ValueError:
                                            pass
                                        s = str(date_str).strip()
                                        if re.match(r"\d{4}-\d{2}-\d{2}$", s):
                                            dt = pd.to_datetime(s, format='%Y-%m-%d')
                                            return dt.strftime('%Y-%m-%d')
                                        if re.match(r"\d{2}-\d{2}-\d{4}$", s):
                                            parts = s.split('-')
                                            a, b, y = int(parts[0]), int(parts[1]), int(parts[2])
                                            if a > 31 or b > 31 or y < 1000:
                                                raise ValueError("Invalid date")
                                            if b > 12:
                                                dt = pd.to_datetime(f"{a}-{b}-{y}", format='%m-%d-%Y')  # MM-DD-YYYY
                                            elif a > 12:
                                                dt = pd.to_datetime(f"{b}-{a}-{y}", format='%m-%d-%Y')  # swap for DD-MM-YYYY
                                            else:
                                                dt = pd.to_datetime(s, dayfirst=True)  # ambiguous, default DD-MM
                                            return dt.strftime('%Y-%m-%d')
                                        dt = pd.to_datetime(s, errors='raise')
                                        return dt.strftime('%Y-%m-%d')
                                    except Exception as e:
                                        row_errors.append((idx + 2, f"Date parse error: {e} for '{date_str}'"))
                                        return None

                                dt_start_str = parse_date(start_date_str)
                                dt_end_str = parse_date(end_date_str) if end_date_str else dt_start_str

                                # Validation
                                if not dt_start_str or not dt_end_str:
                                    continue
                                dt_start = datetime.strptime(dt_start_str, '%Y-%m-%d').date()
                                dt_end = datetime.strptime(dt_end_str, '%Y-%m-%d').date()
                                if dt_end < dt_start:
                                    row_errors.append((idx + 2, "End date before start date"))
                                    continue
                                if reason == "others" and not custom_reason:
                                    row_errors.append((idx + 2, "Custom reason required for 'Others'"))
                                    continue

                                reason_final = custom_reason if reason == "others" else reason.capitalize()  # Capitalize for consistency
                                # Calculate deviation
                                try:
                                    deviation_val = float(row["Avg Deviation (%)"]) if pd.notnull(row["Avg Deviation (%)"]) else 0.0
                                except Exception:
                                    deviation_val = 0.0
                                valid_rows.append({
                                    "plant": plant,
                                    "input_name": equip,
                                    "fault_start_date": dt_start_str,
                                    "fault_end_date": dt_end_str,
                                    "reason": reason_final,
                                    "comment": comment,
                                    "deviation": deviation_val,
                                    "date": dt_start_str
                                })
                                
                            if valid_rows:
                                st.markdown("### Preview of Valid Rows")
                                preview_df = pd.DataFrame(valid_rows)
                                # Format deviation to 3 decimals
                                preview_df["deviation"] = preview_df["deviation"].apply(lambda x: f"{x:.3f}")
                                st.dataframe(preview_df)

                                if st.button("Submit Valid Rows"):
                                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    n_processed = 0
                                    n_failed = 0
                                    failed_rows = []
                                    supabase = get_supabase_client()
                                    for entry in valid_rows:
                                        try:
                                            match = supabase.table("deviation_reasons").select("*") \
                                                .eq("plant", entry["plant"]) \
                                                .eq("input_name", entry["input_name"]) \
                                                .eq("fault_start_date", entry["fault_start_date"]) \
                                                .eq("fault_end_date", entry["fault_end_date"]).execute()
                                            if match.data:
                                                # Update existing
                                                record_id = match.data[0]['id']
                                                old_data = match.data[0]
                                                resp = supabase.table("deviation_reasons").update({
                                                    "reason": entry["reason"],
                                                    "comment": entry["comment"],
                                                    "deviation": entry["deviation"],
                                                    "timestamp": now_str,
                                                    "date": entry["date"]
                                                }).eq("id", record_id).execute()
                                                if hasattr(resp, "error") and resp.error:
                                                    st.error(f"Update error: {resp.error}")
                                                    n_failed += 1
                                                    failed_rows.append(entry)
                                                else:
                                                    # Audit log
                                                    supabase.table("reason_audit_log").insert({
                                                        "action_type": "update",
                                                        "record_id": record_id,
                                                        "old_value": str(old_data),
                                                        "new_value": str({
                                                            "reason": entry["reason"],
                                                            "comment": entry["comment"],
                                                            "deviation": entry["deviation"],
                                                            "timestamp": now_str
                                                        }),
                                                        "timestamp": now_str
                                                    }).execute()
                                                    n_processed += 1
                                            else:
                                                # Insert new
                                                full_entry = entry.copy()
                                                full_entry["timestamp"] = now_str
                                                resp = supabase.table("deviation_reasons").insert(full_entry).execute()
                                                if hasattr(resp, "error") and resp.error:
                                                    st.error(f"Insertion error: {resp.error}")
                                                    n_failed += 1
                                                    failed_rows.append(entry)
                                                else:
                                                    # Audit log
                                                    supabase.table("reason_audit_log").insert({
                                                        "action_type": "insert",
                                                        "record_id": None,
                                                        "old_value": None,
                                                        "new_value": str(full_entry),
                                                        "timestamp": now_str
                                                    }).execute()
                                                    n_processed += 1
                                        except Exception as e:
                                            n_failed += 1
                                            failed_rows.append(entry)
                                            st.warning(f"Failed to process row: {e}")

                                    if n_processed > 0:
                                        st.success(f"✅ Processed {n_processed} records (inserted/updated)!")
                                        st.session_state.upload_key += 1
                                    if n_failed > 0:
                                        st.error(f"Failed to process {n_failed} records. See logs.")
                                    if skipped:
                                        st.info(f"Skipped {len(skipped)} blank rows: rows {skipped}")
                                    if row_errors:
                                        for row_num, err in row_errors:
                                            st.error(f"Row {row_num}: {err}")
                                    if n_processed == 0 and row_errors:
                                        st.error("All rows failed validation.")
                                    # Auto-reset
                                    st.rerun()
                            else:
                                st.warning("No valid rows found in the uploaded file.")
                                if skipped:
                                    st.info(f"Skipped {len(skipped)} blank rows: rows {skipped}")
                                if row_errors:
                                    for row_num, err in row_errors:
                                        st.error(f"Row {row_num}: {err}")
                                    st.error("All rows failed validation.")
                    except Exception as e:
                        st.error(f"Error processing upload: {e}")

                # --- Trend for these equipment ---
                st.markdown("---")
                show_trend = st.checkbox("📈 Show Deviation Trend for Equipment", key="portfolio_show_trend")
                if show_trend:
                    import pandas as pd
                    avg_dev_df['Label'] = avg_dev_df.apply(
                        lambda row: f"{row['plant']} - {row['input_name']} - {row['Avg Deviation (%)']:.3f}%", axis=1
                    )
                    equipment_options = avg_dev_df['Label'].tolist()
                    selected_eq_labels = st.multiselect("Select Equipment(s) for Trend View", equipment_options, key="portfolio_trend_eq")
                    if selected_eq_labels:
                        label_to_plant_eq = {label: label.split(' - ')[:2] for label in equipment_options}
                        date_range = pd.date_range(start=portfolio_date_start, end=portfolio_date_end)
                        fig_trend = go.Figure()
                        for label in selected_eq_labels:
                            plant_equip = label_to_plant_eq[label]
                            plant = plant_equip[0]
                            equip = plant_equip[1]
                            eq_trend_df = df_portfolio[(df_portfolio['plant'] == plant) & (df_portfolio['input_name'] == equip)].copy()
                            eq_trend_df['date'] = pd.to_datetime(eq_trend_df['date'])
                            # Aggregate to one value per date if needed (e.g., mean)
                            eq_trend_df = eq_trend_df.groupby('date')['value'].mean().reindex(date_range).reset_index()
                            eq_trend_df.rename(columns={'index': 'date', 'value': 'Deviation'}, inplace=True)
                            fig_trend.add_trace(go.Scatter(
                                x=eq_trend_df['date'],
                                y=eq_trend_df['Deviation'],
                                mode='lines+markers',
                                name=label
                            ))
                        fig_trend.add_hline(y=0, line_dash="dash", line_color="black")
                        fig_trend.add_hline(y=portfolio_threshold, line_dash="dot", line_color="red")
                        fig_trend.update_layout(
                            title="Deviation Trend for Selected Equipment",
                            xaxis_title="Date",
                            yaxis_title="Deviation (%)",
                            plot_bgcolor='white',
                            height=500,
                            xaxis=dict(autorange=True),
                            yaxis=dict(autorange=True)
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("Select at least one equipment to view the trend.")

                # --- Tag Reason/Comment Section (for these equipment) ---
                st.markdown("---")
                with st.expander("**Add Comment**"):
                    avg_dev_df['Label'] = avg_dev_df.apply(
                        lambda row: f"{row['plant']} - {row['input_name']} - {row['Avg Deviation (%)']:.3f}%", axis=1
                    )
                    tag_eq_options = avg_dev_df['Label'].tolist()
                    selected_tag_labels = st.multiselect("Select Equipment(s) for Tagging", tag_eq_options, key="portfolio_tag_eq")

                    REASON_LIST = get_reasons()
                    selected_reason = st.selectbox("Select Reason*", REASON_LIST, key="portfolio_reason")
                    custom_reason_input = ""
                    if selected_reason == "Others":
                        custom_reason_input = st.text_input("Custom Reason*", key="portfolio_custom_reason")
                    tag_date_range = st.date_input("Fault Date Range", value=(), key="portfolio_tag_daterange")
                    if tag_date_range and len(tag_date_range) == 1:
                        tag_date_range = (tag_date_range[0], tag_date_range[0])
                    tag_comment = st.text_area("Comment (Status/Action)*", placeholder="E.g. Issue resolved, cleaning scheduled...", key="portfolio_tag_comment")

                    disable_submit = (
                        not selected_tag_labels or not tag_date_range or len(tag_date_range) != 2 or
                        (selected_reason == "Others" and not custom_reason_input.strip()) or
                        not tag_comment.strip()
                    )

                    from datetime import datetime

                    supabase = get_supabase_client()

                    if st.button("Submit Reason/Comment", disabled=disable_submit, key="portfolio_submit_reason", type="primary"):
                        fault_start_date, fault_end_date = tag_date_range
                        if fault_end_date < fault_start_date:
                            st.error("Invalid date range.")
                        else:
                            reason_to_store = custom_reason_input.strip() if selected_reason == "Others" else selected_reason
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            insert_count = 0
                            inserts = []
                            for label in selected_tag_labels:
                                parts = label.split(' - ')
                                plant = parts[0]
                                equip = parts[1]
                                # [Requirement 6] Include -100% deviation
                                with duckdb.connect(DB_PATH) as con:
                                    dev_query = """
                                        SELECT AVG(value) FROM dgr_data
                                        WHERE plant = ? AND input_name = ? AND date BETWEEN ? AND ?
                                    """
                                    res = con.execute(dev_query, (plant, equip, fault_start_date, fault_end_date)).fetchone()
                                    deviation_val = res[0] if res and res[0] is not None else 0.0
                                # [Requirement 5] One record per range
                                match = supabase.table("deviation_reasons").select("*").eq("plant", plant).eq("input_name", equip).eq("fault_start_date", str(fault_start_date)).eq("fault_end_date", str(fault_end_date)).execute()
                                data = match.data if match.data else []
                                if data:
                                    # Update
                                    record_id = data[0]['id']
                                    old_data = data[0]
                                    supabase.table("deviation_reasons").update({
                                        "reason": reason_to_store,
                                        "comment": tag_comment.strip(),
                                        "deviation": deviation_val,
                                        "timestamp": now_str,
                                        "date": str(fault_start_date)
                                    }).eq("id", record_id).execute()
                                    # Audit
                                    supabase.table("reason_audit_log").insert({
                                        "action_type": "update",
                                        "record_id": record_id,
                                        "old_value": str(old_data),
                                        "new_value": str({
                                            "reason": reason_to_store,
                                            "comment": tag_comment.strip(),
                                            "deviation": deviation_val,
                                            "timestamp": now_str,
                                            "fault_start_date": str(fault_start_date),
                                            "fault_end_date": str(fault_end_date)
                                        }),
                                        "timestamp": now_str
                                    }).execute()
                                    insert_count += 1
                                else:
                                    # Insert
                                    inserts.append({
                                        "plant": plant,
                                        "date": str(fault_start_date),
                                        "input_name": equip,
                                        "deviation": deviation_val,
                                        "reason": reason_to_store,
                                        "comment": tag_comment.strip(),
                                        "timestamp": now_str,
                                        "fault_start_date": str(fault_start_date),
                                        "fault_end_date": str(fault_end_date)
                                    })
                            if inserts:
                                chunk_size = 50
                                for i in range(0, len(inserts), chunk_size):
                                    chunk = inserts[i:i+chunk_size]
                                    try:
                                        resp = supabase.table("deviation_reasons").insert(chunk).execute()
                                        insert_count += len(resp.data)
                                        # Batch audit log for inserts
                                        audit_inserts = []
                                        for insert in chunk:
                                            audit_inserts.append({
                                                "action_type": "insert",
                                                "record_id": None,
                                                "old_value": None,
                                                "new_value": str(insert),
                                                "timestamp": now_str
                                            })
                                        supabase.table("reason_audit_log").insert(audit_inserts).execute()
                                    except Exception as e:
                                        st.error(f"Chunk insert failed: {e}")
                            st.success(f"Reason/comment tagged for {len(selected_tag_labels)} equipment across ranges ({insert_count} records).")
                            st.session_state["portfolio_submit_success"] = True
                            st.rerun()

                st.markdown("---")

                supabase = get_supabase_client()
                # Fetch main fault_df with retry and pagination
                fault_data = []
                offset = 0
                page_size = 1000
                while True:
                    for attempt in range(3):
                        try:
                            resp = supabase.table("deviation_reasons").select("plant,input_name,reason,fault_start_date,fault_end_date,deviation,comment,date,timestamp").in_("plant", portfolio_plants).lte("fault_start_date", str(portfolio_date_end)).gte("fault_end_date", str(portfolio_date_start)).range(offset, offset + page_size - 1).execute()
                            data = resp.data
                            if not data:
                                break
                            fault_data.extend(data)
                            if len(data) < page_size:
                                break
                            offset += page_size
                            break
                        except Exception as e:
                            st.warning(f"Query attempt {attempt+1} failed: {e}")
                            time.sleep(5)
                            if attempt == 2:
                                st.error("Failed to fetch data after 3 attempts.")
                                fault_df = pd.DataFrame()
                                break
                    if len(data) < page_size:
                        break
                fault_df = pd.DataFrame(fault_data) if fault_data else pd.DataFrame()

                with st.expander("📅 Visual Fault Map", expanded=False):  # expanded=True keeps it open by default
                    if fault_df.empty:
                        st.info("No tagged reasons/comments found for selection.")
                    else:
                        fault_df['Label'] = fault_df['plant']
                        # Expand ranges for plot [Requirement 4] Ensure legend
                        expanded_rows = []
                        for _, row in fault_df.iterrows():
                            start = pd.to_datetime(row['fault_start_date'])
                            end = pd.to_datetime(row['fault_end_date'])
                            for d in pd.date_range(start, end):
                                if portfolio_date_start <= d.date() <= portfolio_date_end:  # Filter to selected date range
                                    new_row = row.copy()
                                    new_row['plot_date'] = d.strftime('%Y-%m-%d')
                                    expanded_rows.append(new_row)
                        expanded_df = pd.DataFrame(expanded_rows)
                        if expanded_df.empty:
                            st.info("No tagged reasons/comments in the selected date range.")
                        else:
                            grouped = expanded_df.groupby(['plant', 'plot_date', 'reason']).size().reset_index(name='count')
                            idx_max = grouped.groupby(['plant', 'plot_date'])['count'].idxmax()
                            expanded_df = grouped.loc[idx_max].reset_index(drop=True)

                            # Limit color map to present reasons to avoid extra legend entries
                            present_reasons = expanded_df['reason'].unique()
                            limited_color_map = {r: REASON_COLOR.get(r, "#888") for r in present_reasons}

                            # Grid-like fault map
                            import plotly.express as px

                            # Get all unique reasons present in your data
                            present_reasons = expanded_df['reason'].unique().tolist()

                            # Combine several Plotly qualitative palettes for many unique colors
                            color_pool = (
                                px.colors.qualitative.Plotly +
                                px.colors.qualitative.D3 +
                                px.colors.qualitative.Dark24 +
                                px.colors.qualitative.Light24 +
                                px.colors.qualitative.Safe +
                                px.colors.qualitative.Alphabet
                            )

                            # If too many reasons, generate even more colors
                            def generate_hex_colors(n):
                                import matplotlib
                                import matplotlib.pyplot as plt
                                cmap = plt.get_cmap('hsv', n)
                                return [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n)]

                            if len(present_reasons) > len(color_pool):
                                color_list = generate_hex_colors(len(present_reasons))
                            else:
                                color_list = color_pool

                            # Final mapping from reason to color
                            reason_color_map = {r: color_list[i % len(color_list)] for i, r in enumerate(present_reasons)}

                            fig = px.scatter(
                                expanded_df,
                                x='plot_date',
                                y='plant',
                                color='reason',
                                color_discrete_map=reason_color_map,   # <--- Use the new map!
                                hover_data={'count': True}
                            )

                            fig.update_traces(marker=dict(size=16, line=dict(width=1, color='black')))
                            fig.update_layout(
                                title="Fault Map: Equipment vs Date",
                                xaxis_title="Date",
                                yaxis_title="Plant",
                                plot_bgcolor='white',
                                height=400 + len(fault_df['Label'].unique()) * 10,
                                showlegend=True,
                                xaxis_tickformat='%Y-%m-%d',
                                legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="right", x=1),
                                xaxis=dict(autorange=True),
                                yaxis=dict(autorange=True)
                            )
                            st.plotly_chart(fig, use_container_width=True)




                        # Remove dynamic legend HTML as per user request

                from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

                st.markdown("#### 📝 Reason/Comment Log (Editable & Downloadable)")

                if "show_comments_portfolio" not in st.session_state:
                    st.session_state["show_comments_portfolio"] = False

                if st.button("Show Comments", key="show_comments_portfolio_btn", type="primary"):
                    st.session_state["show_comments_portfolio"] = True

                if st.session_state["show_comments_portfolio"]:
                    with st.spinner("Loading Reason/Comment Log..."):
                        if not fault_df.empty:
                            log_display_df = fault_df.copy()
                            log_display_df = log_display_df.rename(columns={
                                'plant': 'Plant Name',
                                'input_name': 'Equipment Name',
                                'deviation': '%Deviation',
                                'reason': 'Reason',
                                'comment': 'Comment',
                                'date': 'Date',
                                'fault_start_date': 'Fault Start Date',
                                'fault_end_date': 'Fault End Date',
                                'timestamp': 'Tagged Timestamp'
                            })
                            log_display_df = log_display_df.sort_values(["Fault Start Date", "Tagged Timestamp"], ascending=[False, False])

                            # Download button with history_days input (moved here before computation)
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.download_button("Download Log (Excel)", log_display_df.to_csv(index=False), "fault_log.csv", "text/csv")
                            with col2:
                                history_days = st.number_input("No. of Past Days", min_value=1, max_value=365, value=7, key="history_reason_days")

                            supabase = get_supabase_client()

                            ref_date = pd.to_datetime(portfolio_date_end).date()
                            window_start = ref_date - pd.Timedelta(days=history_days - 1)

                            # Fetch history data for precompute with retry and pagination
                            history_data = []
                            offset = 0
                            page_size = 1000
                            while True:
                                for attempt in range(3):
                                    try:
                                        resp = supabase.table("deviation_reasons").select("plant, input_name, reason, fault_start_date, fault_end_date").in_("plant", portfolio_plants).lte("fault_start_date", str(ref_date)).gte("fault_end_date", str(window_start)).range(offset, offset + page_size - 1).execute()
                                        data = resp.data
                                        if not data:
                                            break
                                        history_data.extend(data)
                                        if len(data) < page_size:
                                            break
                                        offset += page_size
                                        break
                                    except Exception as e:
                                        st.warning(f"History query attempt {attempt+1} failed: {e}")
                                        time.sleep(5)
                                        if attempt == 2:
                                            st.error("Failed to fetch history data after 3 attempts.")
                                            history_fault_df = pd.DataFrame()
                                            break
                                if len(data) < page_size:
                                    break
                            history_fault_df = pd.DataFrame(history_data) if history_data else pd.DataFrame()

                            # Precompute reason days
                            reason_map = defaultdict(lambda: defaultdict(set))  # (plant, equip) => reason => set of days

                            for idx, row in history_fault_df.iterrows():
                                plant = row['plant']
                                equip = row['input_name']
                                reason = row.get('reason', '')
                                if not reason:
                                    continue
                                try:
                                    start = pd.to_datetime(row['fault_start_date']).date()
                                    end = pd.to_datetime(row['fault_end_date']).date()
                                except:
                                    continue
                                range_start = max(start, window_start)
                                range_end = min(end, ref_date)
                                if range_start <= range_end:
                                    for day in pd.date_range(range_start, range_end):
                                        reason_map[(plant, equip)][reason].add(day.strftime("%Y-%m-%d"))

                            # Format %Deviation to 3 decimals
                            log_display_df["%Deviation"] = log_display_df["%Deviation"].apply(lambda x: f"{float(x):.3f}" if pd.notnull(x) else "0.000")

                            # Hide Tagged Timestamp
                            if "Tagged Timestamp" in log_display_df.columns:
                                log_display_df = log_display_df.drop(columns=["Tagged Timestamp"])

                            # Insert History Reason after 'Reason' column
                            insert_pos = log_display_df.columns.get_loc("Reason") + 1
                            history_reason_col = log_display_df.apply(
                                lambda row: ", ".join(f"{r} ({len(days)})" for r, days in sorted(reason_map.get((row['Plant Name'], row['Equipment Name']), {}).items(), key=lambda x: -len(x[1])) if len(days) > 0) or "-",
                                axis=1
                            )
                            log_display_df.insert(insert_pos, "History Reason", history_reason_col)

                            # --- AGGrid Table with column filters (CLEANED) ---
                            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

                            # Remove 'rowColor' column if exists
                            if "rowColor" in log_display_df.columns:
                                log_display_df = log_display_df.drop(columns=["rowColor"])

                            # Ensure %Deviation is float for filtering, even if it comes as string with '%'
                            log_display_df['%Deviation'] = (
                                log_display_df['%Deviation']
                                .astype(str)
                                .str.replace('%', '', regex=False)
                                .astype(float)
                            )

                            # FILTER: Only show those entries where %Deviation ≤ threshold
                            log_display_df = log_display_df[log_display_df['%Deviation'] <= threshold]

                            # Optional: Re-format %Deviation as string with 3 decimals for table display
                            log_display_df['%Deviation'] = log_display_df['%Deviation'].apply(lambda x: f"{x:.3f}")

                            # Reorder and select important columns
                            ordered_cols = [
                                "Plant Name", "Equipment Name", "Date", "%Deviation", "Reason", "History Reason", "Comment",
                                # Add other columns if needed, but keep minimal
                                "Fault Start Date", "Fault End Date"
                            ]
                            log_display_df = log_display_df[[col for col in ordered_cols if col in log_display_df.columns]]

                            # Build grid options with wrapping
                            gb = GridOptionsBuilder.from_dataframe(log_display_df)
                            gb.configure_default_column(filter=True, sortable=True, resizable=True, wrapText=True, autoHeight=True)
                            gb.configure_column("Comment", wrapText=True, autoHeight=True)
                            gb.configure_column("History Reason", wrapText=True, autoHeight=True)
                            gridOptions = gb.build()

                            # Custom CSS (if needed, else remove)
                            st.markdown("""
                            <style>
                                .ag-row-hover { background-color: inherit !important; }
                                .ag-cell-focus { border: none !important; }
                                .ag-header-cell { box-shadow: none !important; }
                                .ag-row { box-shadow: none !important; }
                            </style>
                            """, unsafe_allow_html=True)

                            # Display table
                            AgGrid(
                                log_display_df,
                                gridOptions=gridOptions,
                                enable_enterprise_modules=False,
                                update_mode=GridUpdateMode.NO_UPDATE,
                                fit_columns_on_grid_load=True,
                                allow_unsafe_jscode=True,
                                theme='alpine',
                                height=400
                            )

                            # Note: The per-row expander loop has been removed to avoid redundancy with Visual Summary tab
                        else:
                            st.info("No log entries for this selection.")
                else:
                    st.info("Click 'Show Comments' to load the filtered log.")

# --- TAB 5: VISUAL SUMMARY --- (MIGRATED TO SUPABASE)
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import date

with tab5:
    st.markdown('<div class="section-header"><h3>📊 Visual Summary</h3></div>', unsafe_allow_html=True)
    st.markdown("Fault Distribution & Logs")
    st.divider()

    if not plant_select or not date_start or not date_end:
        st.warning("Please select at least one plant and a date range to continue.")
    else:
        # ---- 1. Load Filtered RCA Table from Supabase ----
        supabase = get_supabase_client()
        # Fetch overlapping faults
        summary_query = supabase.table("deviation_reasons").select("*").in_("plant", plant_select).lte("fault_start_date", str(date_end)).gte("fault_end_date", str(date_start)).execute()
        summary_df = pd.DataFrame(summary_query.data) if summary_query.data else pd.DataFrame()

        if summary_df.empty:
            st.info("No reason/comment data found for selected plants and date range.")
        else:
            # ---- 2. Clean Plant Names for Filters ----
            def clean_plant_name(x):
                return x  # No splitting, keep full name

            summary_df['plant_clean'] = summary_df['plant'].apply(clean_plant_name)

            summary_df = summary_df.rename(columns={'input_name': 'Equipment Name'})

            # New: Stacked bar chart for plant-fault breakdown
            plant_reason_dev = summary_df.groupby(['plant', 'reason'])['deviation'].mean().reset_index()
            plant_reason_dev.columns = ['plant', 'reason', 'Avg Deviation (%)']
            import plotly.express as px

            all_reasons = plant_reason_dev['reason'].unique().tolist()
            # Combine many palettes for extra colors
            color_pool = (
                px.colors.qualitative.Plotly +
                px.colors.qualitative.D3 +
                px.colors.qualitative.Dark24 +
                px.colors.qualitative.Light24 +
                px.colors.qualitative.Safe +
                px.colors.qualitative.Alphabet
            )
            def generate_hex_colors(n):
                import matplotlib
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap('hsv', n)
                return [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n)]

            if len(all_reasons) > len(color_pool):
                color_list = generate_hex_colors(len(all_reasons))
            else:
                color_list = color_pool

            reason_color_map = {r: color_list[i % len(color_list)] for i, r in enumerate(all_reasons)}

            plant_reason_fig = px.bar(
                plant_reason_dev,
                x='plant',
                y='Avg Deviation (%)',
                color='reason',
                title="Plant-Specific Fault Breakdown",
                barmode='stack',
                category_orders={"reason": all_reasons},
                color_discrete_map=reason_color_map
            )



            plant_reason_fig.update_layout(
                xaxis=dict(autorange=True),
                yaxis=dict(autorange=True)
            )
            st.plotly_chart(plant_reason_fig, use_container_width=True)

            # ---- 4. Equipment filter as Plant_Equipment ----
            filtered_df = summary_df
            eq_with_logs_raw = filtered_df[['plant_clean', 'Equipment Name']].drop_duplicates()
            eq_with_logs_raw['equip_display'] = eq_with_logs_raw['plant_clean'] + " - " + eq_with_logs_raw['Equipment Name']
            eq_with_logs_display = ["All Equipment"] + sorted(eq_with_logs_raw['equip_display'].unique())
            equip_display_to_real = dict(zip(eq_with_logs_raw['equip_display'], eq_with_logs_raw['Equipment Name']))
            equipment_filter = st.selectbox("Select Equipment:", eq_with_logs_display, key="vs_log_eq_filter")
            if equipment_filter != "All Equipment":
                filtered_df = filtered_df[filtered_df['Equipment Name'] == equip_display_to_real[equipment_filter]]

            # ---- 5. Reason filter ----
            reasons_present = filtered_df['reason'].unique().tolist()
            reasons_display = ["All Reasons"] + sorted([r for r in reasons_present if r])
            reason_filter = st.selectbox("Select Reason:", reasons_display, key="vs_log_reason_filter")
            if reason_filter != "All Reasons":
                filtered_df = filtered_df[filtered_df['reason'] == reason_filter]

            # ---- 6. Show Comments Button [Requirement 2]
            if "show_comments_vs" not in st.session_state:
                st.session_state["show_comments_vs"] = False

            if st.button("Show Comments", type="primary"):
                st.session_state["show_comments_vs"] = True

            if st.session_state["show_comments_vs"]:
                with st.spinner("Loading Visual Summary Log..."):
                    # ---- 7. Sort log by latest date first ----
                    filtered_df = filtered_df.sort_values(["fault_start_date", "timestamp"], ascending=[False, False]).reset_index(drop=True)

                    # Format deviation to 3 decimals
                    filtered_df["deviation"] = filtered_df["deviation"].apply(lambda x: f"{float(x):.3f}" if pd.notnull(x) else "0.000")

                    # Hide Tagged Timestamp
                    if "timestamp" in filtered_df.columns:
                        filtered_df_for_download = filtered_df.drop(columns=["timestamp"])
                    else:
                        filtered_df_for_download = filtered_df

                    # ---- 8. Bulk delete + download (side by side) ----
                    log_df = filtered_df.copy()
                    log_df["label"] = log_df.apply(
                        lambda row: f"{row['plant_clean']} - {row['Equipment Name']} | {row['fault_start_date']}→{row['fault_end_date']} | {row['reason']}", axis=1
                    )
                    label_to_row = {row["label"]: row for _, row in log_df.iterrows()}
                    selected_labels = st.multiselect(
                        "Select comments to delete (multi-select):", log_df["label"].tolist(), key="vs_multiselect_delete"
                    )
                    col_bulk, col_download = st.columns([1, 2])
                    with col_bulk:
                        if st.button("Delete Selected Comments", disabled=len(selected_labels) == 0, key="vs_delete_bulk", type="secondary"):
                            supabase = get_supabase_client()
                            deleted_count = 0
                            for label in selected_labels:
                                row = label_to_row[label]
                                try:
                                    match = supabase.table("deviation_reasons").select("*").eq("plant", row["plant"]).eq("fault_start_date", row["fault_start_date"]).eq("fault_end_date", row["fault_end_date"]).eq("input_name", row["Equipment Name"]).eq("timestamp", row["timestamp"]).execute()
                                    if match.data:
                                        record_id = match.data[0]['id']
                                        old_data = match.data[0]
                                        supabase.table("deviation_reasons").delete().eq("id", record_id).execute()
                                        # Audit log
                                        supabase.table("reason_audit_log").insert({
                                            "action_type": "delete",
                                            "record_id": record_id,
                                            "old_value": str(old_data),
                                            "new_value": None,
                                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        }).execute()
                                        deleted_count += 1
                                except Exception as e:
                                    st.error(f"Delete failed for {label}: {e}")
                            if deleted_count > 0:
                                st.success(f"Deleted {deleted_count} comment(s) successfully!")
                                st.session_state["show_comments_vs"] = True
                                st.rerun()
                    with col_download:
                        st.download_button(
                            "Download Visual Summary Log (Excel)",
                            filtered_df_for_download.drop(columns=["plant_clean"]).to_csv(index=False),
                            "visual_summary_log.csv",
                            "text/csv"
                        )

                    # ---- 9. Individual log entry (edit/delete) ----
                    REASON_LIST = get_reasons()
                    for idx, row in filtered_df.iterrows():
                        expander_label = f"{row['plant_clean']} - {row['Equipment Name']} | {row['fault_start_date']}→{row['fault_end_date']} | %Dev: {row.get('deviation', '0.000')} | Reason: {row['reason']}"
                        with st.expander(expander_label):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Plant Name:** {row['plant_clean']}")
                                st.write(f"**Equipment:** {row['Equipment Name']}")
                                st.write(f"**Fault Start:** {row['fault_start_date']}")
                                st.write(f"**Fault End:** {row['fault_end_date']}")
                            with col2:
                                st.write(f"**%Deviation:** {row.get('deviation', '0.000')}")
                                st.write(f"**Reason:** {row['reason']}")
                                st.write(f"**Comment:** {row['comment']}")
                                # Timestamp hidden

                            new_reason = st.selectbox(
                                "Edit Reason", REASON_LIST,
                                index=REASON_LIST.index(row['reason']) if row['reason'] in REASON_LIST else len(REASON_LIST) - 1,
                                key=f"edit_vs_reason_{idx}"
                            )
                            custom_reason_vs = ""
                            if new_reason == "Others":
                                custom_reason_vs = st.text_input(
                                    "Custom Reason", row['reason'] if row['reason'] not in REASON_LIST else "", key=f"edit_vs_custom_{idx}"
                                )
                            new_comment = st.text_area("Edit Comment", row["comment"], key=f"edit_vs_comment_{idx}")

                            col1b, col2b = st.columns(2)
                            with col1b:
                                if st.button("Update", key=f"vs_update_{idx}", type="primary"):
                                    if not new_comment.strip() or (new_reason == "Others" and not custom_reason_vs.strip()):
                                        st.error("Reason and Comment are mandatory.")
                                    else:
                                        supabase = get_supabase_client()
                                        reason_final = custom_reason_vs.strip() if new_reason == "Others" else new_reason
                                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        try:
                                            match = supabase.table("deviation_reasons").select("*").eq("plant", row["plant"]).eq("fault_start_date", row["fault_start_date"]).eq("fault_end_date", row["fault_end_date"]).eq("input_name", row["Equipment Name"]).eq("timestamp", row["timestamp"]).execute()
                                            if match.data:
                                                record_id = match.data[0]['id']
                                                old_data = match.data[0]
                                                supabase.table("deviation_reasons").update({
                                                    "reason": reason_final,
                                                    "comment": new_comment.strip(),
                                                    "timestamp": now_str  # Update timestamp on edit
                                                }).eq("id", record_id).execute()
                                                # Audit log
                                                supabase.table("reason_audit_log").insert({
                                                    "action_type": "update",
                                                    "record_id": record_id,
                                                    "old_value": str(old_data),
                                                    "new_value": str({
                                                        "reason": reason_final,
                                                        "comment": new_comment.strip(),
                                                        "timestamp": now_str
                                                    }),
                                                    "timestamp": now_str
                                                }).execute()
                                                st.success(f"Updated successfully! New timestamp: {now_str}")
                                                st.session_state["show_comments_vs"] = True
                                                st.rerun()
                                            else:
                                                st.error("No matching record found for update.")
                                        except Exception as e:
                                            st.error(f"Update failed: {e}")
                            with col2b:
                                if st.button("Delete", key=f"vs_delete_{idx}", type="secondary"):
                                    supabase = get_supabase_client()
                                    try:
                                        match = supabase.table("deviation_reasons").select("*").eq("plant", row["plant"]).eq("fault_start_date", row["fault_start_date"]).eq("fault_end_date", row["fault_end_date"]).eq("input_name", row["Equipment Name"]).eq("timestamp", row["timestamp"]).execute()
                                        if match.data:
                                            record_id = match.data[0]['id']
                                            old_data = match.data[0]
                                            supabase.table("deviation_reasons").delete().eq("id", record_id).execute()
                                            # Audit log
                                            supabase.table("reason_audit_log").insert({
                                                "action_type": "delete",
                                                "record_id": record_id,
                                                "old_value": str(old_data),
                                                "new_value": None,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }).execute()
                                            st.success("Deleted successfully!")
                                            st.session_state["show_comments_vs"] = True
                                            st.rerun()
                                        else:
                                            st.error("No matching record found for delete.")
                                    except Exception as e:
                                        st.error(f"Delete failed: {e}")
            else:
                st.info("Click 'Show Comments' to load the filtered log.")
            if filtered_df.empty:
                st.info("No fault log data available for this filter.")

# --- TAB 6: COMMENT MAP ---
with tab6:
    st.markdown('<div class="section-header"><h3>🗺️ Comment Map</h3></div>', unsafe_allow_html=True)
    if not plant_select or not date_start or not date_end:
        st.warning("Please select at least one plant and a date range to continue.")
    else:
        # Load Filtered RCA Table from Supabase
        supabase = get_supabase_client()
        summary_query = supabase.table("deviation_reasons").select("*").in_("plant", plant_select).lte("fault_start_date", str(date_end)).gte("fault_end_date", str(date_start)).execute()
        summary_df = pd.DataFrame(summary_query.data) if summary_query.data else pd.DataFrame()

        if summary_df.empty:
            st.info("No comment data found for selected plants and date range.")
        else:
            # Calculate for table
            table_rows = []
            for plant in plant_select:
                plant_df = df_portfolio[df_portfolio['plant'] == plant]
                inputs_deviated = (plant_df["value"] <= threshold).sum()
                comments_received = summary_df[summary_df['plant'] == plant].shape[0]
                balance = inputs_deviated - comments_received
                table_rows.append({
                    "Plant Name": plant,
                    "No. of Inputs Deviated": inputs_deviated,
                    "No. of Comments Received": comments_received,
                    "Balance": balance
                })
            table_df = pd.DataFrame(table_rows)
            st.table(table_df)

# --- TAB 7: ADD REASON ---
with tab7:
    st.markdown('<div class="section-header"><h3>⚙️ Add Reason</h3></div>', unsafe_allow_html=True)
    supabase = get_supabase_client()
    REASON_LIST = get_reasons()

    # Auto-populate if empty
    def populate_defaults():
        new_reasons = ["AC Switch off", "AC Current Imbalance", "Cloudy Weather"]
        REASON_COLOR.update({
            "AC Switch off": "#FF5733",
            "AC Current Imbalance": "#C70039",
            "Cloudy Weather": "#900C3F"
        })
        all_defaults = INITIAL_REASON_LIST + new_reasons + ["Others"]
        for r in all_defaults:
            try:
                supabase.table("reasons").insert({"reason_name": r}).execute()
            except:
                pass  # Skip if exists
        st.success("Defaults populated!")
        st.rerun()

    if st.button("Populate Default Reasons", type="secondary"):
        populate_defaults()

    # View All Reasons
    st.markdown("### Current Reasons")
    reasons_df = pd.DataFrame(REASON_LIST, columns=["Reason"])
    st.dataframe(reasons_df)

    # Add a New Reason
    st.markdown("### Add New Reason")
    new_reason = st.text_input("New Reason", key="new_reason_box")
    if st.button("Add Reason", type="primary"):
        if not new_reason.strip():
            st.error("Reason cannot be blank.")
        else:
            try:
                # Fetch existing reasons for case-insensitive check
                response = supabase.table("reasons").select("reason_name").execute()
                existing_reasons = [r['reason_name'].strip().lower() for r in response.data]
                if new_reason.strip().lower() in existing_reasons:
                    st.error("Reason already exists (case-insensitive)!")
                else:
                    supabase.table("reasons").insert({"reason_name": new_reason.strip()}).execute()
                    st.success(f"Added '{new_reason.strip()}' successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error adding reason: {e}. Check Supabase permissions or table schema.")

    # Delete a Reason
    st.markdown("### Delete Reason")
    delete_reason = st.selectbox("Select Reason to Delete", REASON_LIST, key="delete_reason_select")
    if st.button("Delete Reason", type="secondary"):
        if delete_reason == "Others":
            st.error("'Others' cannot be deleted.")
        else:
            try:
                # Check usage
                usage = supabase.table("deviation_reasons").select("count").eq("reason", delete_reason).execute().data[0]['count']
                if usage > 0:
                    st.warning(f"This reason is used in {usage} past records. Deleting will remove it from future options but not update history.")
                supabase.table("reasons").delete().eq("reason_name", delete_reason).execute()
                st.success(f"Deleted '{delete_reason}' successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting reason: {e}. Check Supabase permissions or table schema.")

    # Edit a Reason
    st.markdown("### Edit Reason")
    edit_reason = st.selectbox("Select Reason to Edit", REASON_LIST, key="edit_reason_select")
    updated_reason = st.text_input("Updated Reason", value=edit_reason, key="updated_reason_box")
    if st.button("Update Reason", type="primary"):
        if not updated_reason.strip():
            st.error("Reason cannot be blank.")
        elif updated_reason in REASON_LIST and updated_reason != edit_reason:
            st.error("Updated reason already exists.")
        else:
            try:
                # Update in reasons table
                supabase.table("reasons").update({"reason_name": updated_reason.strip()}).eq("reason_name", edit_reason).execute()
                # Update historical records
                supabase.table("deviation_reasons").update({"reason": updated_reason.strip()}).eq("reason", edit_reason).execute()
                st.success(f"Updated '{edit_reason}' to '{updated_reason.strip()}' successfully, including historical records!")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating reason: {e}. Check Supabase permissions or table schema.")