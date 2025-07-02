import streamlit as st
import pandas as pd
import duckdb
from datetime import datetime
import plotly.graph_objects as go
import base64  # <--- Required for watermark

DB_PATH = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
MAPPING_SHEET = "Mapping Sheet.xlsx"

st.set_page_config(page_title="DGR Deviation Dashboard", layout="wide")

# ------ WATERMARK LOGO BACKGROUND (paste here) ------
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
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: 530px 300px;
    background-repeat: no-repeat;
    background-position: center 120px;
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
# ------ END WATERMARK ------

st.title("📊 DGR Deviation Dashboard")
st.markdown("Analyze inverter/block-wise deviation across plants using data from DuckDB.")

# --- Load Mapping Sheet ---
@st.cache_data
def load_mapping():
    return pd.read_excel(MAPPING_SHEET)

# --- Load Mapping Sheet ---
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

# ------- TABS -----------
tab1, tab2 = st.tabs(["Generate Table", "Generate Ranking"])

with tab1:
    if st.button("Generate Table"):
        with st.spinner("Processing..."):
            with duckdb.connect(DB_PATH) as con:
                # Pull all data in range for selected plants
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

            # --- SINGLE PLANT MODE ---
            if len(plant_select) == 1:
                plant_name = plant_select[0]
                mapping_row = mapping_df[mapping_df["Plant_Name"] == plant_name].iloc[0]
                start_col = mapping_row["Data_Start_Col"]
                end_col = mapping_row["Data_End_Col"]
                equipment_names = df["input_name"].unique().tolist()
                # Only take mapped columns for this plant (from mapping)
                if start_col in equipment_names and end_col in equipment_names:
                    sidx = equipment_names.index(start_col)
                    eidx = equipment_names.index(end_col)
                    mapped_equipment = equipment_names[sidx:eidx+1]
                else:
                    mapped_equipment = equipment_names

                table_rows = []
                serial = 1
                for eq in mapped_equipment:
                    flagged = df[(df["input_name"] == eq) & (df["value"] <= threshold)]
                    if not flagged.empty:
                        avg_deviation = flagged["value"].mean()
                        num_days = flagged["date"].nunique()
                        table_rows.append([
                            serial, plant_name, eq, f"{avg_deviation:.2f}%", num_days
                        ])
                        serial += 1
                # Display
                outdf = pd.DataFrame(table_rows, columns=[
                    "Serial No.", "Plant_Name", "Equipment_Name", "%Deviation", "No. of Days Deviated"
                ])
                if outdf.empty:
                    st.info("No deviations below threshold found for this plant in selected range.")
                else:
                    st.markdown("### Equipment-wise Deviation Table")
                    st.dataframe(outdf, use_container_width=True)

            # --- MULTI-PLANT MODE ---
            else:
                rows = []
                serial = 1
                for plant in df['plant'].unique():
                    plant_df = df[df['plant'] == plant]
                    # Total non-blank cells in selected range (denominator)
                    total_input_cells = plant_df['value'].notnull().sum()
                    # All cells ≤ threshold (numerator)
                    inputs_deviated = (plant_df['value'] <= threshold).sum()
                    # Normalised Deviation %
                    norm_deviation = 100 * inputs_deviated / total_input_cells if total_input_cells > 0 else 0
                    # Average deviation for flagged cells (optional)
                    avg_deviation = plant_df[plant_df['value'] <= threshold]['value'].mean() if inputs_deviated > 0 else 0

                    rows.append([
                        serial,
                        plant,
                        f"{avg_deviation:.2f}%",
                        inputs_deviated,
                        total_input_cells,
                        f"{norm_deviation:.2f}%"
                    ])
                    serial += 1

                outdf = pd.DataFrame(rows, columns=[
                    "Serial No.", "Plant_Name", "%Deviation", "No. of Inputs Deviated", "Total Inputs", "Normalised Deviation (%)"
                ])
                if outdf.empty:
                    st.info("No deviations below threshold found for any plant in selected range.")
                else:
                    st.markdown("### Plant-wise Deviation Summary")
                    st.dataframe(outdf, use_container_width=True)
with tab2:
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
            ranked = ranked.sort_values("Normalised Deviation (%)").reset_index(drop=True)
            ranked['Rank'] = ranked.index + 1

            st.markdown("### 🌟 Plant Normalised Deviation Ranking")
            st.dataframe(ranked, use_container_width=True)

            # --- CHART CODE STARTS HERE ---
            import matplotlib
            import numpy as np

            cmap = matplotlib.colormaps['RdYlGn_r']
            max_dev = ranked["Normalised Deviation (%)"].max()
            min_dev = ranked["Normalised Deviation (%)"].min()
            if max_dev == min_dev:
                norm = np.ones(len(ranked)) * 0.5
            else:
                norm = (ranked["Normalised Deviation (%)"] - min_dev) / (max_dev - min_dev)
            bar_colors = [matplotlib.colors.rgb2hex(cmap(x)) for x in norm]

            text_labels = [f"Rank {row.Rank}, {row['Normalised Deviation (%)']:.2f}%" for idx, row in ranked.iterrows()]

            fig = go.Figure(go.Bar(
                y=ranked["Plant"],
                x=ranked["Normalised Deviation (%)"],
                orientation='h',
                marker_color=bar_colors,
                text=text_labels,
                textposition='outside',
                insidetextanchor='end',
                hovertemplate="Plant: %{y}<br>Deviation: %{x:.2f}%<extra></extra>"
            ))

            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                title="🏆 Plant Ranking by Normalised Deviation (Lower is Better)",
                xaxis_title="Normalised Deviation (%)",
                yaxis_title="Plant",
                font=dict(family="Segoe UI, Arial", size=16),
                xaxis=dict(tickformat=".2f"),
                plot_bgcolor='white',
                margin=dict(t=70, l=150, r=40),
                height=100 + 60 * len(ranked),
            )
            st.plotly_chart(fig, use_container_width=True)
