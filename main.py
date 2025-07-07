import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
import base64
import matplotlib
import numpy as np

from visualisation_tab import render_visualisation_tab  # <-- import the modular tab

DB_PATH = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
MAPPING_SHEET = "Mapping Sheet.xlsx"

st.set_page_config(page_title="DGR Deviation Dashboard", layout="wide")

# ------ WATERMARK LOGO BACKGROUND ------
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

# --- Threshold ---
if "threshold" not in st.session_state:
    st.session_state["threshold"] = -3.0
threshold = st.number_input("Deviation Threshold (e.g., -3 means ≤ -3%)",
                            value=st.session_state["threshold"], step=0.1, key="threshold")

# --- Date Range ---
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

tab1, tab2, tab3 = st.tabs(["Generate Table", "Generate Ranking", "Visualisation"])

# --- TAB 1: GENERATE TABLE ---
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
                        if value is not None:
                            table_rows.append([
                                serial, plant_name, eq, f"{value:.2f}%", 1, count
                            ])
                            serial += 1
                    outdf = pd.DataFrame(table_rows, columns=[
                        "Serial No.", "Plant_Name", "Equipment_Name", "%Deviation", "No. of Days (shown: 1)", threshold_label
                    ])
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
                    st.dataframe(outdf, use_container_width=True)
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
                    st.dataframe(outdf, use_container_width=True)

# --- TAB 2: GENERATE RANKING ---
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
                        table_rows.append([eq, value if value is not None else 0, 1, count])
                    outdf = pd.DataFrame(table_rows, columns=["Equipment Name", "Deviation", "No. of Days", threshold_label])
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
                outdf["Deviation"] = outdf["Deviation"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
                outdf[threshold_label] = outdf[threshold_label].astype(int).fillna(0)

                st.markdown("### 🌟 Equipment-wise Deviation Ranking")
                st.dataframe(outdf, use_container_width=True)

                dev_numeric = [float(str(x).replace('%','')) for x in outdf["Deviation"] if x not in ["-", None]]
                equipment_names = outdf["Equipment Name"].tolist()

                bar_colors = []
                if dev_numeric:
                    min_dev = min(dev_numeric)
                    max_dev = max(dev_numeric)
                    for v in dev_numeric:
                        if v < 0:
                            norm = min(abs(v) / max(abs(min_dev), 1), 1)
                            bar_colors.append(matplotlib.colors.rgb2hex(matplotlib.colormaps['Reds'](0.4 + 0.6*norm)))
                        elif v > 0:
                            norm = min(v / max(max_dev, 1), 1)
                            bar_colors.append(matplotlib.colors.rgb2hex(matplotlib.colormaps['Greens'](0.4 + 0.6*norm)))
                        else:
                            bar_colors.append('#ffff66')
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
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    title="🏆 Equipment Ranking by Deviation (Red = Worst, Green = Best)",
                    xaxis_title="Deviation (%)",
                    yaxis_title="Equipment",
                    font=dict(family="Segoe UI, Arial", size=16),
                    xaxis=dict(tickformat=".2f"),
                    plot_bgcolor='white',
                    margin=dict(t=70, l=150, r=40),
                    height=100 + 60 * len(outdf),
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
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

                st.markdown("### 🌟 Plant Normalised Deviation Ranking (Lower ABS = Better, Higher ABS = Redder)")
                st.dataframe(ranked.drop(columns="AbsDeviation"), use_container_width=True)

                abs_devs = ranked["AbsDeviation"]
                dev_numeric = ranked["Normalised Deviation (%)"].tolist()
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
                text_labels = [f"Rank {row.Rank}, {row['Normalised Deviation (%)']:.2f}%" for idx, row in ranked.iterrows()]
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
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    title="🏆 Plant Ranking by |Normalised Deviation| (Lower ABS = Greener, Higher ABS = Redder)",
                    xaxis_title="Normalised Deviation (%)",
                    yaxis_title="Plant",
                    font=dict(family="Segoe UI, Arial", size=16),
                    xaxis=dict(tickformat=".2f"),
                    plot_bgcolor='white',
                    margin=dict(t=70, l=150, r=40),
                    height=100 + 60 * len(ranked),
                )
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: VISUALISATION ---
with tab3:
    render_visualisation_tab(plant_select, date_start, date_end, threshold)
