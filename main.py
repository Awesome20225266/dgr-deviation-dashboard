import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
import base64
import matplotlib
import numpy as np

DB_PATH = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
MAPPING_SHEET = "Mapping Sheet.xlsx"

st.set_page_config(page_title="DGR Deviation Dashboard", layout="wide")

# ------ WATERMARK LOGO BACKGROUND (centered, before UI) ------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = "JSW-Energy-completes-acquisition-of-4.7-GW-renewable-energy-platform-from-O2-Power.jpg"
 # Update as needed!
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

st.title("📊 DGR Deviation Dashboard")
st.markdown("Analyze inverter/block-wise deviation across plants using data from DuckDB.")

@st.cache_data
def load_mapping():
    return pd.read_excel(MAPPING_SHEET)

mapping_df = load_mapping()
plants_available = mapping_df["Plant_Name"].unique().tolist()

# --- Select All Plants Option ---
select_all = st.checkbox("Select All Plants", value=True)
if select_all:
    default_plants = plants_available
else:
    default_plants = []

plant_select = st.multiselect("Select Plant(s)", options=plants_available, default=default_plants)

if not plant_select:
    st.warning("Please select at least one plant.")
    st.stop()

threshold = st.number_input("Deviation Threshold (e.g., -3 means ≤ -3%)", value=-3.0, step=0.1)

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
    date_range = st.date_input(
        "Select Date Range",
        (date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.warning("Please select a start and end date (date range)!")
        st.stop()
    date_start, date_end = date_range
else:
    st.warning("No data found in the database for selected plants.")
    st.stop()

tab1, tab2 = st.tabs(["Generate Table", "Generate Ranking"])

with tab1:
    if st.button("Generate Table"):
        with st.spinner("Processing..."):
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

            # --- SINGLE PLANT MODE ---
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
                for eq in mapped_equipment:
                    flagged = df[
                        (df["input_name"] == eq)
                        & (df["value"] <= threshold)
                        & (df["value"] != 100)
                    ]
                    if not flagged.empty:
                        avg_deviation = flagged["value"].mean()
                        num_days = flagged["date"].nunique()
                        table_rows.append([
                            serial, plant_name, eq, f"{avg_deviation:.2f}%", num_days
                        ])
                        serial += 1
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
                    total_input_cells = plant_df['value'].notnull().sum()
                    inputs_deviated = (plant_df['value'] <= threshold).sum()
                    norm_deviation = 100 * inputs_deviated / total_input_cells if total_input_cells > 0 else 0
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

            # ---- EQUIPMENT-WISE RANKING FOR SINGLE PLANT ----
            if len(plant_select) == 1:
                plant_name = plant_select[0]
                equipment_names = df["input_name"].unique().tolist()
                table_rows = []
                for eq in equipment_names:
                    flagged = df[
                        (df["input_name"] == eq)
                        & (df["value"] <= threshold)
                        & (df["value"] != 100)
                    ]
                    if not flagged.empty:
                        avg_deviation = flagged["value"].mean()
                        num_days = flagged["date"].nunique()
                        table_rows.append([eq, avg_deviation, num_days])
                outdf = pd.DataFrame(table_rows, columns=["Equipment Name", "%Deviation", "No. of Days Deviated"])
                # SORT: Most negative (worst) at top, least negative (best) at bottom
                outdf = outdf.sort_values("%Deviation").reset_index(drop=True)
                outdf["%Deviation"] = outdf["%Deviation"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
                st.markdown("### 🌟 Equipment-wise Deviation Ranking")
                st.dataframe(outdf, use_container_width=True)

                dev_numeric = [float(x.replace('%','')) for x in outdf["%Deviation"] if x != "-"]
                fig = go.Figure()
                if dev_numeric:
                    cmap = matplotlib.colormaps['RdYlGn']
                    min_dev = min(dev_numeric)
                    max_dev = max(dev_numeric)
                    if min_dev == max_dev:
                        norm = np.ones(len(dev_numeric)) * 1.0
                    else:
                        norm = 1 - (np.array(dev_numeric) - min_dev) / (max_dev - min_dev)
                    bar_colors = [matplotlib.colors.rgb2hex(cmap(x)) for x in norm]
                    fig = go.Figure(go.Bar(
                        y=outdf["Equipment Name"],
                        x=dev_numeric,
                        orientation='h',
                        marker_color=bar_colors,
                        text=outdf["%Deviation"],
                        textposition='outside',
                        insidetextanchor='end',
                        hovertemplate="Equipment: %{y}<br>Deviation: %{x:.2f}%<extra></extra>"
                    ))
                    fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    title="🏆 Equipment Ranking by Avg. Deviation (Lower is Worse)",
                    xaxis_title="Avg. Deviation (%)",
                    yaxis_title="Equipment",
                    font=dict(family="Segoe UI, Arial", size=16),
                    xaxis=dict(tickformat=".2f"),
                    plot_bgcolor='white',
                    margin=dict(t=70, l=150, r=40),
                    height=100 + 60 * len(outdf),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---- PLANT-WISE RANKING FOR MULTI-PLANT ----
            else:
                rows = []
                for plant in df['plant'].unique():
                    plant_df = df[df['plant'] == plant]
                    total_input_cells = plant_df['value'].notnull().sum()
                    inputs_deviated = (plant_df['value'] <= threshold).sum()
                    norm_deviation = 100 * inputs_deviated / total_input_cells if total_input_cells > 0 else 0
                    rows.append([plant, norm_deviation])
                ranked = pd.DataFrame(rows, columns=["Plant", "Normalised Deviation (%)"])
                # SORT: Highest deviation at top (red), lowest at bottom (green)
                ranked = ranked.sort_values("Normalised Deviation (%)", ascending=False).reset_index(drop=True)
                ranked['Rank'] = ranked.index + 1

                st.markdown("### 🌟 Plant Normalised Deviation Ranking")
                st.dataframe(ranked, use_container_width=True)

                dev_numeric = ranked["Normalised Deviation (%)"].tolist()
                fig = go.Figure()
                if dev_numeric:
                    cmap = matplotlib.colormaps['RdYlGn']
                    min_dev = min(dev_numeric)
                    max_dev = max(dev_numeric)
                    if min_dev == max_dev:
                        norm = np.ones(len(dev_numeric)) * 1.0
                    else:
                        norm = 1 - (np.array(dev_numeric) - min_dev) / (max_dev - min_dev)
                    bar_colors = [matplotlib.colors.rgb2hex(cmap(x)) for x in norm]
                    text_labels = [f"Rank {row.Rank}, {row['Normalised Deviation (%)']:.2f}%" for idx, row in ranked.iterrows()]
                    fig = go.Figure(go.Bar(
                        y=ranked["Plant"],
                        x=dev_numeric,
                        orientation='h',
                        marker_color=bar_colors,
                        text=text_labels,
                        textposition='outside',
                        insidetextanchor='end',
                        hovertemplate="Plant: %{y}<br>Deviation: %{x:.2f}%<extra></extra>"
                    ))
                    fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    title="🏆 Plant Ranking by Normalised Deviation (Higher is Worse, Redder)",
                    xaxis_title="Normalised Deviation (%)",
                    yaxis_title="Plant",
                    font=dict(family="Segoe UI, Arial", size=16),
                    xaxis=dict(tickformat=".2f"),
                    plot_bgcolor='white',
                    margin=dict(t=70, l=150, r=40),
                    height=100 + 60 * len(ranked),
                )
                st.plotly_chart(fig, use_container_width=True)
