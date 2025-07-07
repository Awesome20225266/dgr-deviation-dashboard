import streamlit as st
import pandas as pd
import duckdb
from datetime import datetime
import plotly.graph_objects as go
import matplotlib
import numpy as np

DB_PATH = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
MAPPING_SHEET = "Mapping Sheet.xlsx"

st.set_page_config(page_title="Plant Ranking Dashboard", layout="wide")
st.title("🏆 Plant Normalised Deviation Ranking")
st.markdown("See plant rankings by Absolute Normalised Deviation (%). Lowest ABS = best (green), highest ABS = worst (red)!")

@st.cache_data
def load_mapping():
    return pd.read_excel(MAPPING_SHEET)

mapping_df = load_mapping()
plants_available = mapping_df["Plant_Name"].unique().tolist()
plant_select = st.multiselect("Select Plant(s)", options=plants_available, default=plants_available)
if "threshold" not in st.session_state:
    st.session_state["threshold"] = -3.0
threshold = st.number_input("Deviation Threshold (e.g., -3 means ≤ -3%)",
                            value=st.session_state["threshold"], step=0.1, key="threshold")

with duckdb.connect(DB_PATH) as con:
    date_query = f"""
        SELECT DISTINCT date FROM {TABLE_NAME}
        WHERE plant IN ({','.join(['?'] * len(plant_select))})
        ORDER BY date
    """
    dates_result = con.execute(date_query, plant_select).fetchall()
    all_dates = [d[0] for d in dates_result]

if all_dates:
    date_min = min(all_dates)
    date_max = max(all_dates)
    date_start, date_end = st.date_input("Select Date Range", (date_min, date_max), min_value=date_min, max_value=date_max)
else:
    st.warning("No data found in the database for selected plants.")
    st.stop()

if st.button("Generate Ranking"):
    with st.spinner("Ranking..."):
        with duckdb.connect(DB_PATH) as con:
            query = f"""
                SELECT plant, date, input_name, value
                FROM {TABLE_NAME}
                WHERE plant IN ({','.join(['?'] * len(plant_select))})
                  AND date BETWEEN ? AND ?
            """
            params = plant_select + [date_start, date_end]
            df = con.execute(query, params).df()
        if df.empty:
            st.warning("No data found in selected range.")
            st.stop()

        if len(plant_select) == 1:
            # Equipment-wise for one plant
            equipment_names = df["input_name"].unique().tolist()
            table_rows = []
            threshold_label = f"No. of days ≤ {threshold}%"
            for eq in equipment_names:
                eq_df = df[df["input_name"] == eq]
                avg_dev = eq_df["value"].mean() if not eq_df.empty else 0
                n_days = eq_df["date"].nunique()
                count = (eq_df["value"] <= threshold).sum() if not eq_df.empty else 0
                table_rows.append([eq, avg_dev, n_days, count])
            outdf = pd.DataFrame(table_rows, columns=["Equipment Name", "Deviation", "No. of Days", threshold_label])
            outdf = outdf.sort_values("Deviation", ascending=True).reset_index(drop=True)
            outdf["Rank"] = outdf.index + 1
            outdf["Deviation"] = outdf["Deviation"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
            outdf[threshold_label] = outdf[threshold_label].astype(int).fillna(0)
            st.markdown("### 🌟 Equipment-wise Deviation Ranking")
            st.dataframe(outdf, use_container_width=True)
        else:
            # Plant-wise for multiple plants
            rows = []
            for plant in df['plant'].unique():
                plant_df = df[df['plant'] == plant]
                total_input_cells = plant_df['value'].notnull().sum()
                inputs_deviated = (plant_df['value'] <= threshold).sum()
                norm_deviation = 100 * inputs_deviated / total_input_cells if total_input_cells > 0 else 0
                rows.append([plant, norm_deviation, inputs_deviated])
            ranked = pd.DataFrame(rows, columns=["Plant", "Normalised Deviation (%)", f"No. of Inputs ≤ {threshold}%"])

            ranked["AbsDeviation"] = ranked["Normalised Deviation (%)"].abs()
            ranked = ranked.sort_values("AbsDeviation").reset_index(drop=True)
            ranked['Rank'] = ranked.index + 1

            st.markdown("### 🌟 Plant Normalised Deviation Ranking (by Absolute Value)")
            st.dataframe(ranked.drop(columns="AbsDeviation"), use_container_width=True)

        # --- Plot logic unchanged (if you want to keep bar plot add here) ---

