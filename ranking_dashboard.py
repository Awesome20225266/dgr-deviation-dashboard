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
threshold = st.number_input("Deviation Threshold (e.g., -3 means ≤ -3%)", value=-3.0, step=0.1)

# Fetch available dates from DB for selected plants
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
        rows = []
        for plant in df['plant'].unique():
            plant_df = df[df['plant'] == plant]
            total_input_cells = plant_df['value'].notnull().sum()
            inputs_deviated = (plant_df['value'] <= threshold).sum()
            norm_deviation = 100 * inputs_deviated / total_input_cells if total_input_cells > 0 else 0
            rows.append([plant, norm_deviation])
        ranked = pd.DataFrame(rows, columns=["Plant", "Normalised Deviation (%)"])

        # Sort and rank by ABS value
        ranked = ranked.reindex(ranked["Normalised Deviation (%)"].abs().sort_values().index)
        ranked = ranked.reset_index(drop=True)
        ranked['Rank'] = ranked.index + 1

        st.markdown("### 🌟 Plant Normalised Deviation Ranking (by Absolute Value)")
        st.dataframe(ranked, use_container_width=True)

        # Color by ABS (green = best, red = worst)
        abs_devs = ranked["Normalised Deviation (%)"].abs()
        norm = (abs_devs - abs_devs.min()) / (abs_devs.max() - abs_devs.min()) if len(abs_devs) > 1 else abs_devs
        cmap = matplotlib.colormaps['RdYlGn_r']
        bar_colors = [matplotlib.colors.rgb2hex(cmap(x)) for x in norm]

        text_labels = [f"Rank {row.Rank}, {row['Normalised Deviation (%)']:.2f}%" for idx, row in ranked.iterrows()]

        fig = go.Figure(go.Bar(
            y=ranked["Plant"],  # Y-axis: plant names
            x=ranked["Normalised Deviation (%)"],  # X-axis: deviation %
            orientation='h',
            marker_color=bar_colors,
            text=text_labels,
            textposition='outside',
            insidetextanchor='end',
            hovertemplate="Plant: %{y}<br>Deviation: %{x:.2f}%<extra></extra>"
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title="Plant Ranking by |Normalised Deviation| (Lower ABS = Better, Higher ABS = Redder)",
            xaxis_title="Normalised Deviation (%)",
            yaxis_title="Plant",
            font=dict(family="Segoe UI, Arial", size=16),
            xaxis=dict(tickformat=".2f"),
            plot_bgcolor='white',
            margin=dict(t=70, l=150, r=40),
            height=100 + 60 * len(ranked),
        )
        st.plotly_chart(fig, use_container_width=True)
