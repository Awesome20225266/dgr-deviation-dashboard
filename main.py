import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
import base64
import matplotlib
import numpy as np
from datetime import datetime

from visualisation_tab import render_visualisation_tab
import io
import pandas as pd

def to_excel_bytes(df, reason_options):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Deviation Data')
        workbook  = writer.book
        worksheet = writer.sheets['Deviation Data']
        # Find the "Reason" column index
        reason_col = df.columns.get_loc("Reason")
        # Apply dropdown to all rows in Reason column (row 2 to N+1 in Excel)
        worksheet.data_validation(
            first_row=1, first_col=reason_col,
            last_row=len(df), last_col=reason_col,
            options={
                'validate': 'list',
                'source': reason_options,
                'input_message': 'Select a reason',
                'error_message': 'Choose only from list'
            }
        )
    output.seek(0)
    return output.getvalue()

DB_PATH = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
MAPPING_SHEET = "Mapping Sheet.xlsx"

# -- Ensure deviation_reasons table has all columns
with duckdb.connect(DB_PATH) as con:
    con.execute("ALTER TABLE deviation_reasons ADD COLUMN IF NOT EXISTS deviation DOUBLE;")
    con.execute("ALTER TABLE deviation_reasons ADD COLUMN IF NOT EXISTS fault_start_date TEXT;")
    con.execute("ALTER TABLE deviation_reasons ADD COLUMN IF NOT EXISTS fault_end_date TEXT;")

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

tab1, tab2, tab3, tab4, tab5 = st.tabs([ 
    "Generate Table", "Generate Ranking", "Visualisation", "Portfolio Deep Analytics", "Visual Summary"
])
REASON_OPTIONS = [
    "Soiling", "Shadow", "Disconnected String", "Connector Burn", "Fuse Failure", "IGBT Failure",
        "Module Damage", "Power Clipping", "Vegetation Growth", "Bypass diode", "Degradation", "Temperature Loss",
        "RISO Fault", "MPPT Malfunction", "Grid Outage", "Load Curtailment", "Efficiency loss", "Ground Fault",
        "Module Mismatch", "IIGBT Issue", "Array Misalignment", "Tracker Failure", "Inverter Fan Issue",
        "Bifacial factor Loss", "Power Limitation", "Others"
]
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

# --- TAB 4: PORTFOLIO DEEP ANALYTICS ---
# [Due to token limits, paste the full working Tab 4 code from my earlier responses here]

# --- TAB 5: VISUAL SUMMARY ---
# [Due to token limits, paste the full working Tab 5 code from my earlier responses here]
from visualisation_tab import REASON_LIST

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import duckdb
import streamlit as st

with tab4:
    st.subheader("📊 Portfolio Deep Analytics (Uniform Calculation)")

    # Use main selectors
    portfolio_plants = plant_select
    portfolio_threshold = threshold
    portfolio_date_start = date_start
    portfolio_date_end = date_end

    if not portfolio_plants:
        st.warning("Please select at least one plant.")
        st.stop()

    with st.spinner("Fetching data..."):
        with duckdb.connect(DB_PATH) as con:
            query = f"""
                SELECT plant, date, input_name, value
                FROM {TABLE_NAME}
                WHERE plant IN ({','.join(['?'] * len(portfolio_plants))})
                  AND date BETWEEN ? AND ?
            """
            params = portfolio_plants + [portfolio_date_start, portfolio_date_end]
            df_portfolio = con.execute(query, params).df()

    if df_portfolio.empty:
        st.warning("No data found for selected criteria.")
        st.stop()

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
        st.stop()

    # --- Bar chart coloring and layout ---
    import plotly.express as px

    plant_list = avg_dev_df['plant'].unique().tolist()
    px_palette = px.colors.qualitative.Plotly
    plant_color_map = {plant: px_palette[i % len(px_palette)] for i, plant in enumerate(plant_list)}

    # Assign color per bar by plant
    bar_colors = avg_dev_df['plant'].map(plant_color_map)

    fig = go.Figure(
        go.Bar(
            x=avg_dev_df['input_name'],
            y=avg_dev_df['Avg Deviation (%)'],
            marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in avg_dev_df['Avg Deviation (%)']],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Deviation: %{y:.2f}%<extra></extra>",
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
    )

    # Visually group by plant (optional: shaded background rectangles)
    curr_x = 0
    for plant in plant_list:
        count = (avg_dev_df['plant'] == plant).sum()
    # x0 = start bar index - 0.5, x1 = end bar index + 0.5 (no overlap)
        fig.add_vrect(
            x0=curr_x-0.5,
            x1=curr_x+count-1+0.5,   # notice the -1!
            fillcolor=plant_color_map[plant],
            opacity=0.08,
            line_width=0
    )
        curr_x += count


    # Add custom legend
    legend_items = [
        f"<span style='color:{plant_color_map[plant]}; font-weight:bold;'>&#11044;</span> {plant} ({(avg_dev_df['plant'] == plant).sum()})"
        for plant in plant_list
    ]
    fig.update_layout(
        margin=dict(t=70, l=60, r=20, b=120),
        annotations=[
            dict(
                text="<br>".join(legend_items),
                align='left',
                showarrow=False,
                xref="paper", yref="paper",
                x=1.01, y=1, xanchor='left', yanchor='top',
                bordercolor="#ccc",
                borderwidth=1,
                bgcolor="white"
            )
        ]
    )

    
    


    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_hline(y=portfolio_threshold, line_dash="dot", line_color="red")
    N = len(avg_dev_df)

    fig.update_layout(
        title="Equipment Deviation by Plant (Avg of All Days, Only < Threshold)",
        xaxis_title="Equipment",
        yaxis_title="Avg Deviation (%)",
        barmode='group',
        plot_bgcolor='white',
        legend_title_text="Plant (Equipment Count)",
        height=500,
        xaxis_tickangle=-45 if N > 15 else 0,
        bargap=0.05,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    
        # --- Download Button (same uniform logic) ---
    download_df = avg_dev_df.rename(columns={"plant": "Plant Name", "input_name": "Equipment Name"})
    download_df["Avg Deviation (%)"] = download_df["Avg Deviation (%)"].round(2)
    # Add empty columns for upload
    download_df["Fault Start Date"] = ""
    download_df["Fault End Date"] = ""
    download_df["Reason"] = ""
    download_df["Custom Reason"] = ""
    download_df["Comment"] = ""

    st.download_button(
    label="Download Deviation Data (Excel)",
    data=to_excel_bytes(download_df, REASON_LIST),
    file_name=f"Portfolio_Avg_Deviations_{portfolio_date_start}_{portfolio_date_end}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    # >>>>>>>>>>>>  PASTE THE NEW BULK UPLOAD CODE BELOW THIS LINE <<<<<<<<<<<<<
    import io
    import pandas as pd
    import duckdb
    from datetime import datetime

    REASON_LIST = [
        "Soiling", "Shadow", "Disconnected String", "Connector Burn", "Fuse Failure", "IGBT Failure",
        "Module Damage", "Power Clipping", "Vegetation Growth", "Bypass diode", "Degradation", "Temperature Loss",
        "RISO Fault", "MPPT Malfunction", "Grid Outage", "Load Curtailment", "Efficiency loss", "Ground Fault",
        "Module Mismatch", "IIGBT Issue", "Array Misalignment", "Tracker Failure", "Inverter Fan Issue",
        "Bifacial factor Loss", "Power Limitation", "Others"
    ]
    
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
        _Only rows with Reason and Comment will be processed. Timestamp is automatic._

        **Note:** For date range, all dates in range will be tagged in DB. If only Fault Start Date is filled, only that date will be tagged.
    """)

        uploaded_file = st.file_uploader(
            "Upload filled deviation file (xls, xlsx, csv)", 
            type=['xls', 'xlsx', 'csv'], 
            key="bulk_upload_deviation"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)

                # Clean columns and check
                required_cols = [
                    "Plant Name", "Equipment Name", "Avg Deviation (%)",
                    "Fault Start Date", "Fault End Date", "Reason", "Custom Reason", "Comment"
                ]
                missing_cols = [c for c in required_cols if c not in df_upload.columns]
                if missing_cols:
                    st.error(f"Missing columns in upload: {missing_cols}")
                else:
                    # Only process rows with Reason and Comment
                    valid = df_upload["Reason"].astype(str).str.strip().ne("") & df_upload["Comment"].astype(str).str.strip().ne("")
                    df_upload = df_upload[valid].copy()
                    if df_upload.empty:
                        st.warning("No valid rows (Reason + Comment) found to upload.")
                    else:
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        n_added = 0
                        with duckdb.connect(DB_PATH) as con:
                            for idx, row in df_upload.iterrows():
                                plant = row["Plant Name"]
                                equip = row["Equipment Name"]
                                deviation = row.get("Avg Deviation (%)", None)
                                reason = str(row["Reason"]).strip()
                                custom_reason = str(row.get("Custom Reason", "")).strip()
                                comment = str(row["Comment"]).strip()
                                start_date = str(row["Fault Start Date"]).strip()
                                end_date = str(row["Fault End Date"]).strip() or start_date
                                # Sanity: dates
                                try:
                                    dt_start = pd.to_datetime(start_date).date()
                                    dt_end = pd.to_datetime(end_date).date() if end_date else dt_start
                                except Exception:
                                    st.warning(f"Date format error in row {idx+2} ({plant}/{equip}) – skipped.")
                                    continue
                                reason_final = custom_reason if reason.lower() == "others" and custom_reason else reason
                                # For all dates in range
                                for d in pd.date_range(dt_start, dt_end):
                                    # DB dedup check: don't allow exact same record for same plant/equipment/date/reason/comment
                                    exists = con.execute("""
                                        SELECT COUNT(*) FROM deviation_reasons
                                        WHERE plant=? AND input_name=? AND date=? AND reason=? AND comment=?
                                    """, [plant, equip, str(d.date()), reason_final, comment]).fetchone()[0]
                                    if exists == 0:
                                        con.execute("""
                                            INSERT INTO deviation_reasons 
                                            (plant, date, input_name, deviation, reason, comment, timestamp, fault_start_date, fault_end_date)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """, [
                                            plant, str(d.date()), equip,
                                            deviation if not pd.isnull(deviation) else None,
                                            reason_final, comment, now_str,
                                            str(dt_start), str(dt_end)
                                        ])
                                        n_added += 1
                        st.success(f"✅ Uploaded and added {n_added} reason records!")
                        st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing upload: {e}")


        # --- Trend for these equipment ---
    st.markdown("---")
    show_trend = st.checkbox("📈 Show Deviation Trend for Equipment", key="portfolio_show_trend")
    if show_trend:
        import pandas as pd
        avg_dev_df['Label'] = avg_dev_df.apply(
            lambda row: f"{row['plant']}_{row['input_name']}_{row['Avg Deviation (%)']:.2f}%", axis=1
    )
        equipment_options = avg_dev_df['Label'].tolist()
        selected_eq_labels = st.multiselect("Select Equipment(s) for Trend View", equipment_options, key="portfolio_trend_eq")
        if selected_eq_labels:
            label_to_plant_eq = {label: (row['plant'], row['input_name']) for label, row in zip(avg_dev_df['Label'], avg_dev_df.to_dict('records'))}
            date_range = pd.date_range(start=portfolio_date_start, end=portfolio_date_end)
            fig_trend = go.Figure()
            for label in selected_eq_labels:
                plant, equip = label_to_plant_eq[label]
                eq_trend_df = df_portfolio[
                    (df_portfolio['plant'] == plant) &
                    (df_portfolio['input_name'] == equip)
                ].copy()
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
                height=500
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Select at least one equipment to view the trend.")


    # --- Tag Reason/Comment Section (for these equipment) ---
    st.markdown("---")
    st.markdown("### 🏷️ Tag Reason/Comment for Underperforming Equipment")
    avg_dev_df['Label'] = avg_dev_df.apply(
        lambda row: f"{row['plant']}_{row['input_name']}_{row['Avg Deviation (%)']:.2f}%", axis=1
    )
    tag_eq_options = avg_dev_df['Label'].tolist()
    selected_tag_labels = st.multiselect("Select Equipment(s) for Tagging", tag_eq_options, key="portfolio_tag_eq")

    selected_reason = st.selectbox("Select Reason*", REASON_LIST, key="portfolio_reason")
    custom_reason_input = ""
    if selected_reason == "Others":
        custom_reason_input = st.text_input("Custom Reason*", key="portfolio_custom_reason")
    tag_date_range = st.date_input("Fault Date Range", value=(portfolio_date_start, portfolio_date_end), key="portfolio_tag_daterange")
    tag_comment = st.text_area("Comment (Status/Action)*", placeholder="E.g. Issue resolved, cleaning scheduled...", key="portfolio_tag_comment")

    disable_submit = (
        not selected_tag_labels or
        (selected_reason == "Others" and not custom_reason_input.strip()) or
        not tag_comment.strip()
    )

    if st.button("Submit Reason/Comment", disabled=disable_submit, key="portfolio_tag_submit"):
        if not tag_date_range:
            tag_start, tag_end = portfolio_date_end, portfolio_date_end
        else:
            tag_start, tag_end = tag_date_range[0], tag_date_range[1]

        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entries = []
        for label in selected_tag_labels:
            plant, equip, deviation_str = label.rsplit('_', 2)
            deviation_val = float(deviation_str.replace('%', ''))
            current_date = tag_start
            while current_date <= tag_end:
                entries.append({
                    "plant": plant,
                    "date": str(current_date),
                    "input_name": equip,
                    "deviation": deviation_val,
                    "reason": custom_reason_input.strip() if selected_reason == "Others" else selected_reason,
                    "comment": tag_comment.strip(),
                    "timestamp": timestamp_now,
                    "fault_start_date": str(tag_start),
                    "fault_end_date": str(tag_end)
                })
                current_date += pd.Timedelta(days=1)
        with duckdb.connect(DB_PATH) as con:
            for entry in entries:
                con.execute(f"""
                    INSERT INTO deviation_reasons (plant, date, input_name, deviation, reason, comment, timestamp, fault_start_date, fault_end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, list(entry.values()))
        st.success(f"✅ Saved {len(entries)} records successfully!")
        st.rerun()
    
    # --- Fault Map & Editable Log below as before ---
    # ... (use your existing code for visual fault map, log, etc. The key thing is ALL upstream logic is now uniform.)


    # ---- Visual Fault Map ----
    st.markdown("---")
    st.markdown("#### 📅 Visual Fault Map")
    with duckdb.connect(DB_PATH) as con:
        fault_query = f"""
            SELECT plant, date, input_name, deviation, reason, comment, fault_start_date, fault_end_date, timestamp
            FROM deviation_reasons
            WHERE plant IN ({','.join(['?'] * len(portfolio_plants))})
              AND CAST(date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        """
        fault_params = portfolio_plants + [portfolio_date_start, portfolio_date_end]
        fault_df = con.execute(fault_query, fault_params).df()

    if fault_df.empty:
        st.info("No tagged reasons/comments found for selection.")
    else:
        fault_df['Label'] = fault_df.apply(lambda r: f"{r['plant']}_{r['input_name']}", axis=1)
        fault_df['Date'] = pd.to_datetime(fault_df['date'])
        reason_colors = {r: f"#{np.random.randint(0, 0xFFFFFF):06x}" for r in fault_df['reason'].unique()}
        scatter_x, scatter_y, scatter_color, scatter_text = [], [], [], []
        for _, row in fault_df.iterrows():
            scatter_x.append(row['Date'])
            scatter_y.append(row['Label'])
            scatter_color.append(reason_colors[row['reason']])
            scatter_text.append(
                f"Date: {row['Date'].date()}<br>Plant: {row['plant']}<br>Equipment: {row['input_name']}<br>Deviation: {row['deviation']:.2f}%<br>Reason: {row['reason']}<br>Comment: {row['comment']}"
            )
        fault_fig = go.Figure(go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            marker=dict(color=scatter_color, size=16, line=dict(width=1, color='black')),
            text=scatter_text,
            hoverinfo='text'
        ))
        fault_fig.update_layout(
            title="Fault Map: Equipment vs Date",
            xaxis_title="Date",
            yaxis_title="Equipment",
            plot_bgcolor='white',
            height=400 + len(fault_df['Label'].unique()) * 10
        )
        st.plotly_chart(fault_fig, use_container_width=True)

    # ---- Editable Log Table ----
    st.markdown("#### 📝 Reason/Comment Log (Editable & Downloadable)")
    if not fault_df.empty:
        log_display_df = fault_df.copy()
        for col in ['Label', 'Date']:
            if col in log_display_df.columns:
                log_display_df = log_display_df.drop(columns=[col])
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
        log_display_df = log_display_df.sort_values(["Date", "Tagged Timestamp"], ascending=[False, False])
        st.dataframe(log_display_df, use_container_width=True)
        st.download_button("Download Log (Excel)", log_display_df.to_csv(index=False), "fault_log.csv", "text/csv")
    else:
        st.info("No fault log data available.")



import duckdb
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import date

with tab5:
    st.subheader("📊 Visual Summary: Fault Distribution & Logs")

    # ---- 1. Load Full RCA Table (all time) ----
    with duckdb.connect(DB_PATH) as con:
        summary_df = con.execute("""
            SELECT plant, date, input_name, deviation, reason, comment, fault_start_date, fault_end_date, timestamp
            FROM deviation_reasons
        """).df()

    if summary_df.empty:
        st.info("No reason/comment data found for any plant.")
        st.stop()

    # ---- 2. Clean Plant Names for Filters ----
    def clean_plant_name(x):
        # Adjust splitting logic if your plant naming has a different pattern!
        return x.split('_')[0].strip() if '_' in x else x.strip()
    summary_df['plant_clean'] = summary_df['plant'].apply(clean_plant_name)

    summary_df = summary_df.rename(columns={'input_name': 'Equipment Name'})

    # ---- 3. Plant filter with "All Plants" ----
    plants_with_logs = sorted(summary_df['plant_clean'].unique())
    plants_with_logs_display = ["All Plants"] + plants_with_logs
    plant_filter = st.selectbox("Select Plant:", plants_with_logs_display, key="vs_log_plant_filter")

    # ---- 4. Date range filter for Pie Chart (uses min/max from filtered plant) ----
    df_for_pie = summary_df if plant_filter == "All Plants" else summary_df[summary_df['plant_clean'] == plant_filter]
    min_date = pd.to_datetime(df_for_pie['date']).min().date()
    max_date = pd.to_datetime(df_for_pie['date']).max().date()
    pie_date_range = st.date_input("Select Date Range for Pie Chart", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    pie_start, pie_end = pie_date_range

    # ---- 5. Pie Chart (responsive to plant + date) ----
    pie_df = df_for_pie[
        (pd.to_datetime(df_for_pie['date']).dt.date >= pie_start) &
        (pd.to_datetime(df_for_pie['date']).dt.date <= pie_end)
    ]
    pie_title = f"Reason Distribution for {plant_filter}"
    if (pie_start != min_date) or (pie_end != max_date):
        pie_title += f", {pie_start} to {pie_end}"
    reason_counts = pie_df['reason'].value_counts().reset_index()
    reason_counts.columns = ['Reason', 'Count']
    pie_fig = go.Figure(go.Pie(
        labels=reason_counts['Reason'],
        values=reason_counts['Count'],
        hole=0.4
    ))
    pie_fig.update_layout(title=pie_title, height=450)
    st.plotly_chart(pie_fig, use_container_width=True)

    # ---- 6. Equipment filter as Plant_Equipment ----
    filtered_df = summary_df if plant_filter == "All Plants" else summary_df[summary_df['plant_clean'] == plant_filter]
    eq_with_logs_raw = filtered_df[['plant_clean', 'Equipment Name']].drop_duplicates()
    eq_with_logs_raw['equip_display'] = eq_with_logs_raw['plant_clean'] + "_" + eq_with_logs_raw['Equipment Name']
    eq_with_logs_display = ["All Equipment"] + sorted(eq_with_logs_raw['equip_display'].unique())
    equip_display_to_real = dict(zip(eq_with_logs_raw['equip_display'], eq_with_logs_raw['Equipment Name']))
    equipment_filter = st.selectbox("Select Equipment:", eq_with_logs_display, key="vs_log_eq_filter")
    if equipment_filter != "All Equipment":
        filtered_df = filtered_df[filtered_df['Equipment Name'] == equip_display_to_real[equipment_filter]]

    # ---- 7. Reason filter ----
    reasons_present = filtered_df['reason'].unique().tolist()
    reasons_display = ["All Reasons"] + sorted([r for r in reasons_present if r])
    reason_filter = st.selectbox("Select Reason:", reasons_display, key="vs_log_reason_filter")
    if reason_filter != "All Reasons":
        filtered_df = filtered_df[filtered_df['reason'] == reason_filter]

    # ---- 8. Sort log by latest date first ----
    filtered_df = filtered_df.sort_values(["date", "timestamp"], ascending=[False, False]).reset_index(drop=True)

    # ---- 9. Bulk delete + download (side by side) ----
    filtered_df["label"] = filtered_df.apply(
        lambda row: f"{row['plant_clean']}_{row['Equipment Name']} | {row['date']} | {row['reason']} | {row['fault_start_date']}→{row['fault_end_date']}", axis=1
    )
    label_to_row = {row["label"]: row for _, row in filtered_df.iterrows()}
    selected_labels = st.multiselect(
        "Select comments to delete (multi-select):", filtered_df["label"].tolist(), key="vs_multiselect_delete"
    )
    col_bulk, col_download = st.columns([1, 2])
    with col_bulk:
        if st.button("Delete Selected Comments", disabled=len(selected_labels) == 0, key="vs_delete_bulk"):
            with duckdb.connect(DB_PATH) as con:
                for label in selected_labels:
                    row = label_to_row[label]
                    con.execute(
                        "DELETE FROM deviation_reasons WHERE plant=? AND date=? AND input_name=? AND timestamp=?",
                        (row["plant"], row["date"], row["Equipment Name"], row["timestamp"])
                    )
            st.success(f"Deleted {len(selected_labels)} comment(s).")
            st.rerun()
    with col_download:
        st.download_button(
            "Download Visual Summary Log (Excel)",
            filtered_df.drop(columns=["label", "plant_clean"]).to_csv(index=False),
            "visual_summary_log.csv",
            "text/csv"
        )

    # ---- 10. Individual log entry (edit/delete) ----
    reason_options = [
        "Soiling", "Shadow", "Disconnected String", "Connector Burn", "Fuse Failure", "IGBT Failure",
        "Module Damage", "Power Clipping", "Vegetation Growth", "Bypass diode", "Degradation", "Temperature Loss",
        "RISO Fault", "MPPT Malfunction", "Grid Outage", "Load Curtailment", "Efficiency loss", "Ground Fault",
        "Module Mismatch", "IIGBT Issue", "Array Misalignment", "Tracker Failure", "Inverter Fan Issue",
        "Bifacial factor Loss", "Power Limitation", "Others"
    ]
    for idx, row in filtered_df.iterrows():
        expander_label = (
            f"{row['plant_clean']}_{row['Equipment Name']} | {row['date']} | "
            f"Fault: {row['fault_start_date']}→{row['fault_end_date']} | "
            f"%Dev: {row['deviation']} | Reason: {row['reason']}"
        )
        with st.expander(expander_label):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Plant Name:** {row['plant_clean']}")
                st.write(f"**Equipment:** {row['Equipment Name']}")
                st.write(f"**Date:** {row['date']}")
                st.write(f"**Fault Start:** {row['fault_start_date']}")
                st.write(f"**Fault End:** {row['fault_end_date']}")
            with col2:
                st.write(f"**%Deviation:** {row['deviation']}")
                st.write(f"**Reason:** {row['reason']}")
                st.write(f"**Comment:** {row['comment']}")
                st.write(f"**Timestamp:** {row['timestamp']}")

            new_reason = st.selectbox(
                "Edit Reason", reason_options,
                index=reason_options.index(row['reason']) if row['reason'] in reason_options else len(reason_options) - 1,
                key=f"edit_vs_reason_{idx}"
            )
            custom_reason_vs = ""
            if new_reason == "Others":
                custom_reason_vs = st.text_input(
                    "Custom Reason", row['reason'] if row['reason'] not in reason_options else "", key=f"edit_vs_custom_{idx}"
                )
            new_comment = st.text_area("Edit Comment", row["comment"], key=f"edit_vs_comment_{idx}")

            col1b, col2b = st.columns(2)
            with col1b:
                if st.button("Update", key=f"vs_update_{idx}"):
                    if not new_comment.strip() or (new_reason == "Others" and not custom_reason_vs.strip()):
                        st.error("Reason and Comment are mandatory.")
                    else:
                        with duckdb.connect(DB_PATH) as con:
                            con.execute(f"""
                                UPDATE deviation_reasons
                                SET reason=?, comment=?
                                WHERE plant=? AND date=? AND input_name=? AND timestamp=?
                            """, (
                                custom_reason_vs.strip() if new_reason == "Others" else new_reason,
                                new_comment.strip(),
                                row["plant"], row["date"], row["Equipment Name"], row["timestamp"]
                            ))
                        st.success("Updated!")
                        st.rerun()
            with col2b:
                if st.button("Delete", key=f"vs_delete_{idx}"):
                    with duckdb.connect(DB_PATH) as con:
                        con.execute(f"""
                            DELETE FROM deviation_reasons
                            WHERE plant=? AND date=? AND input_name=? AND timestamp=?
                        """, (row["plant"], row["date"], row["Equipment Name"], row["timestamp"]))
                    st.success("Deleted!")
                    st.rerun()

    if filtered_df.empty:
        st.info("No fault log data available for this filter.")
