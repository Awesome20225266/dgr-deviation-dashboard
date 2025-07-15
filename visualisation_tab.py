# visualisation_tab.py (complete updated script)

import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client
from functools import lru_cache
import plotly.express as px
from postgrest import APIError as PostgrestAPIError

SUPABASE_URL = "https://ubkcxehguactwwcarkae.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVia2N4ZWhndWFjdHd3Y2Fya2FlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMTU3OTYsImV4cCI6MjA2Nzc5MTc5Nn0.NPiJj_o-YervOE1dPxWRJhEI1fUwxT3Dptz-JszChLo"

def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

DB_PATH = "dgr_data.duckdb"
PERF_TABLE = "dgr_data"

INITIAL_REASON_LIST = [
    "Soiling", "Shadow", "Disconnected String", "Connector Burn", "Fuse Failure", "IGBT Failure", "Module Damage",
    "Power Clipping", "Vegetation Growth", "Bypass diode", "Degradation", "Temperature Loss", "RISO Fault",
    "MPPT Malfunction", "Grid Outage", "Load Curtailment", "Efficiency loss", "Ground Fault", "Module Mismatch",
    "IIGBT Issue", "Array Misalignment", "Tracker Failure", "Inverter Fan Issue",
    "Bifacial factor Loss",
    "Power Limitation"
]

REASON_COLOR = {
    "Soiling": "#FFA500", "Shadow": "#999966", "Disconnected String": "#f44336", "Connector Burn": "#ef5350",
    "Fuse Failure": "#9c27b0", "IGBT Failure": "#ce93d8", "Module Damage": "#607d8b", "Power Clipping": "#ff9800",
    "Vegetation Growth": "#4caf50", "Bypass diode": "#673ab7", "Degradation": "#607d8b", "Temperature Loss": "#bdbdbd",
    "RISO Fault": "#2196f3", "MPPT Malfunction": "#00bcd4", "Grid Outage": "#03a9f4", "Load Curtailment": "#cddc39",
    "Efficiency loss": "#009688", "Ground Fault": "#795548", "Module Mismatch": "#607d8b", "IIGBT Issue": "#ff5252",
    "Array Misalignment": "#ffe082", "Tracker Failure": "#b71c1c", "Inverter Fan Issue": "#0277bd",
    "Bifacial factor Loss": "#81c784", "Power Limitation": "#ff7043", "Others": "#616161"
}

def get_reasons():
    try:
        supabase = get_supabase_client()
        response = supabase.table("reasons").select("reason_name").execute()
        reasons = [r['reason_name'] for r in response.data]
        reasons.sort()
        if "Others" not in reasons:
            reasons.append("Others")
        return reasons
    except PostgrestAPIError as e:
        if e.code == '42P01':
            return sorted(INITIAL_REASON_LIST + ["Others"])
        else:
            raise e
    except Exception as e:
        return sorted(INITIAL_REASON_LIST + ["Others"])

@lru_cache(maxsize=128)
def fetch_perf_data(plant_tuple, date_start_str, date_end_str):
    plant_list = list(plant_tuple)
    with duckdb.connect(DB_PATH) as con:
        perf_query = f"""
            SELECT * FROM {PERF_TABLE}
            WHERE plant IN ({','.join(['?']*len(plant_list))})
              AND date BETWEEN ? AND ?
        """
        perf_params = plant_list + [date_start_str, date_end_str]
        return con.execute(perf_query, perf_params).df()

def render_visualisation_tab(plant_select, date_start, date_end, threshold):
    REASON_LIST = get_reasons()

    # Reset session states if submit happened (use a flag)
    if "submit_success" not in st.session_state:
        st.session_state["submit_success"] = False
    if st.session_state["submit_success"]:
        st.session_state["reason_equip_select"] = None
        st.session_state["reason_main_select"] = None
        st.session_state["fault_date_range"] = ()
        st.session_state["reason_comment_box"] = ""
        st.session_state["submit_success"] = False  # Reset flag

    # --- 1. Initialize session state (must be before widget definition!) ---
    for key, default in [
        ("reason_main_select", None),
        ("custom_reason_box", ""),
        ("reason_comment_box", "")
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # [Requirement 1] Check for plant and date selection
    if not plant_select or not date_start or not date_end:
        st.info("Please select plants and date range to view visualisation.")
        return

    st.header("🌞 Visualisation & Root Cause Analytics")

    date_start_str = str(date_start)
    date_end_str = str(date_end)
    perf_df = fetch_perf_data(tuple(plant_select), date_start_str, date_end_str)

    tabs = st.tabs(["Plant Overview", "Equipment Drilldown", "Root Cause Analytics"])

    # ---------- Plant Overview Tab ----------
    with tabs[0]:
        st.subheader("Plant Performance Comparison")
        if perf_df.empty:
            st.info("No performance data found for selection.")
        else:
            # Multi-plant
            if len(plant_select) > 1:
                results = []
                for plant, group in perf_df.groupby("plant"):
                    total_inputs = len(group)
                    deviated = group[group["value"] < threshold]
                    num_deviated = len(deviated)
                    avg_dev = deviated["value"].mean() if num_deviated else 0
                    score = (num_deviated / total_inputs) * avg_dev if total_inputs else 0
                    results.append({
                        "Plant": plant,
                        "Score": score,
                        "Inputs Deviated": num_deviated,
                        "Total Inputs": total_inputs,
                        "Avg Deviation": avg_dev
                    })
                plant_score_df = pd.DataFrame(results).sort_values("Score")
                max_abs_score = max(abs(plant_score_df["Score"].min()), abs(plant_score_df["Score"].max()), 1)
                colors = [
                    f'rgba(200,40,40,{min(1, abs(s)/max_abs_score)})' if s < 0 else f'rgba(50,160,90,{min(1, abs(s)/max_abs_score)})'
                    for s in plant_score_df["Score"]
                ]
                fig = go.Figure(go.Bar(
                    x=plant_score_df["Score"],
                    y=plant_score_df["Plant"],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{x:.2f}" for x in plant_score_df["Score"]],
                    textposition='outside',
                    hovertemplate=(
                        "Plant: %{y}<br>"
                        "Score: %{x:.2f}<br>"
                        "Inputs Deviated: %{customdata[0]}<br>"
                        "Avg Deviation: %{customdata[1]:.2f}%"
                    ),
                    customdata=np.stack((plant_score_df["Inputs Deviated"], plant_score_df["Avg Deviation"]), axis=-1)
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                fig.update_layout(
                    title="Plant Comparison Score (Negative is Worse)",
                    xaxis_title="Comparison Score",
                    yaxis_title="Plant",
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(plant_score_df)
            elif len(plant_select) == 1 and str(date_start) == str(date_end):
                plant_name = plant_select[0]
                equip_df = perf_df[perf_df["plant"] == plant_name]
                if equip_df.empty:
                    st.info("No data for selected plant and date.")
                else:
                    equip_df = equip_df.sort_values("value")
                    colors = ['red' if v < 0 else 'green' for v in equip_df["value"]]
                    fig = go.Figure(go.Bar(
                        x=equip_df["input_name"],
                        y=equip_df["value"],
                        marker_color=colors,
                        text=[f"{v:.2f}" for v in equip_df["value"]],
                        textposition='outside'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    fig.update_layout(
                        title=f"Equipment Deviation on {date_start} ({plant_name})",
                        xaxis_title="Equipment",
                        yaxis_title="Deviation (%)",
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(equip_df[["input_name", "value"]].rename(columns={"value": "Deviation (%)"}))
            elif len(plant_select) == 1:
                plant_name = plant_select[0]
                trend_df = perf_df[perf_df["plant"] == plant_name]
                if trend_df.empty:
                    st.info("No data for selected plant and date range.")
                else:
                    date_scores = []
                    for dt, group in trend_df.groupby("date"):
                        total_inputs = len(group)
                        deviated = group[group["value"] < threshold]
                        num_deviated = len(deviated)
                        avg_dev = deviated["value"].mean() if num_deviated else 0
                        score = (num_deviated / total_inputs) * avg_dev if total_inputs else 0
                        date_scores.append({"Date": dt, "Score": score})
                    date_scores_df = pd.DataFrame(date_scores).sort_values("Date")
                    colors = ['red' if s < 0 else 'green' for s in date_scores_df["Score"]]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=date_scores_df["Date"],
                        y=date_scores_df["Score"],
                        marker_color=colors,
                        name="Score per Day"
                    ))
                    fig.add_trace(go.Scatter(
                        x=date_scores_df["Date"],
                        y=date_scores_df["Score"],
                        mode="lines+markers",
                        marker=dict(color='black', size=8),
                        name="Trend"
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    fig.update_layout(
                        title=f"Plant Score Trend: {plant_name}",
                        xaxis_title="Date",
                        yaxis_title="Score",
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(date_scores_df.rename(columns={"Score": "Comparison Score"}))
            else:
                st.info("Please select at least one plant and a valid date range.")

    # ----------- Equipment Drilldown Tab ----------- (MIGRATED TO SUPABASE FOR REASONS/COMMENTS)
    with tabs[1]:
        if len(plant_select) != 1:
            st.info("Please select exactly one plant to view equipment drilldown.")
            return

        st.subheader(f"Equipment Drilldown: {plant_select[0]}")
        plant_df = perf_df[perf_df["plant"] == plant_select[0]]
        if plant_df.empty:
            st.info("No equipment data for selected plant and range.")
            return

        threshold_value = float(threshold)
        def get_bar_color(val):
            if val >= 0:
                return "green"
            elif val < threshold_value:
                return "red"
            else:
                return "orange"
        equip_metrics = (
            plant_df.groupby("input_name")["value"]
            .mean()
            .reset_index(name="Avg Deviation")
        )
        colors = [get_bar_color(v) for v in equip_metrics["Avg Deviation"]]

        st.markdown(
            f"<b>Bar color legend:</b> "
            f"<span style='color:green'>Green</span>: ≥ 0%   "
            f"<span style='color:orange'>Orange</span>: 0% to {threshold_value}%   "
            f"<span style='color:red'>Red</span>: < {threshold_value}%",
            unsafe_allow_html=True
        )

        fig = go.Figure(go.Bar(
            x=equip_metrics["input_name"],
            y=equip_metrics["Avg Deviation"],
            marker_color=colors,
            text=[f"{v:.2f}" for v in equip_metrics["Avg Deviation"]],
            textposition='outside'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_hline(y=threshold_value, line_dash="dot", line_color="red")
        fig.update_layout(
            title=f"Avg Deviation by Equipment ({plant_select[0]})",
            xaxis_title="Equipment",
            yaxis_title="Avg Deviation (%)",
            plot_bgcolor='white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Emoji color-coded dropdown for equipment selection ---
        def emoji_label(row):
            v = row['Avg Deviation']
            if v >= 0:
                return f"🟩 {row['input_name']} ({v:.2f}%)"
            elif v < threshold_value:
                return f"🟥 {row['input_name']} ({v:.2f}%)"
            else:
                return f"🟧 {row['input_name']} ({v:.2f}%)"
        equip_metrics['emoji_label'] = equip_metrics.apply(emoji_label, axis=1)
        emoji_to_name = dict(zip(equip_metrics['emoji_label'], equip_metrics['input_name']))

        selected_equipment = st.multiselect(
            "Show Trend for Equipment (select one or more to compare)",
            equip_metrics['emoji_label'].tolist(),
            key="multi_trend_eq_select"
        )
        selected_equipment_names = [emoji_to_name[label] for label in selected_equipment]

        if selected_equipment_names:
            trend_fig = go.Figure()
            color_palette = [
                "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
                "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
                "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"
            ]
            for idx, eq in enumerate(selected_equipment_names):
                eq_trend_df = plant_df[plant_df["input_name"] == eq].sort_values("date")
                trend_fig.add_trace(go.Scatter(
                    x=eq_trend_df["date"],
                    y=eq_trend_df["value"],
                    mode='lines+markers',
                    name=eq,
                    line=dict(color=color_palette[idx % len(color_palette)], width=2),
                    marker=dict(size=7)
                ))
            trend_fig.add_hline(y=0, line_dash="dash", line_color="black")
            trend_fig.update_layout(
                title="Deviation Trend by Equipment",
                xaxis_title="Date", yaxis_title="Deviation (%)",
                plot_bgcolor='white'
            )
            st.plotly_chart(trend_fig, use_container_width=True)

        st.markdown("---")
        with st.expander("**Add Comment**"):
            eq_dropdown_options = [f"🌐 Whole Plant"] + list(equip_metrics['emoji_label'])
            selected_label = st.selectbox("Select Equipment (with status badge)", eq_dropdown_options, key="reason_equip_select")
            selected_eq = None
            if selected_label:
                if selected_label.startswith("🌐"):
                    selected_eq = "Whole Plant"
                else:
                    selected_eq = selected_label.split(' ', 1)[1].split(' (')[0]

            reason = st.selectbox("Select Reason*", REASON_LIST, key="reason_main_select")
            if reason == "Others":
                custom_reason = st.text_input("Custom Reason*", key="custom_reason_box")
            else:
                custom_reason = ""

            # ----- NEW: Unified Fault Date Range Picker -----
            fault_date_range = st.date_input(
                "Fault Date Range",
                value=(),   # Enable range picker with empty initial [fixed for range selection]
                key="fault_date_range"
            )
            if fault_date_range and len(fault_date_range) == 1:
                fault_date_range = (fault_date_range[0], fault_date_range[0])  # Handle single date

            comment = st.text_area(
                "Comment (Action/Status)*",
                placeholder="E.g. Issue resolved, cleaning scheduled, replacement pending...",
                key="reason_comment_box"
            )

            submit_disabled = (
                (not selected_eq) or not fault_date_range or
                (reason == "Others" and not custom_reason.strip()) or
                (reason != "Others" and not reason) or
                (not comment.strip())
            )

            supabase = get_supabase_client()

            if st.button("Submit Reason/Comment", disabled=submit_disabled, key="equip_submit_reason"):
                fault_start_date, fault_end_date = fault_date_range if len(fault_date_range) == 2 else (fault_date_range[0], fault_date_range[0])
                if fault_end_date < fault_start_date:
                    st.error("End date cannot be before start date.")
                else:
                    reason_to_store = custom_reason.strip() if reason == "Others" else reason
                    insert_count = 0
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # [Requirement 5] One record per range
                    # Calculate average deviation over range
                    with duckdb.connect(DB_PATH) as con:
                        if selected_eq != "Whole Plant":
                            dev_query = """
                                SELECT AVG(value) FROM dgr_data
                                WHERE plant = ? AND input_name = ? AND date BETWEEN ? AND ?
                            """
                            res = con.execute(dev_query, (plant_select[0], selected_eq, fault_start_date, fault_end_date)).fetchone()
                            deviation_val = res[0] if res and res[0] is not None else 0.0
                        else:
                            dev_query = """
                                SELECT AVG(value) FROM dgr_data
                                WHERE plant = ? AND date BETWEEN ? AND ?
                            """
                            res = con.execute(dev_query, (plant_select[0], fault_start_date, fault_end_date)).fetchone()
                            deviation_val = res[0] if res and res[0] is not None else 0.0
                    # Check existence for update
                    match = supabase.table("deviation_reasons").select("*").eq("plant", plant_select[0]).eq("input_name", selected_eq).eq("fault_start_date", str(fault_start_date)).eq("fault_end_date", str(fault_end_date)).execute()
                    if match.data:
                        # Update
                        record_id = match.data[0]['id']
                        old_data = match.data[0]
                        supabase.table("deviation_reasons").update({
                            "reason": reason_to_store,
                            "comment": comment.strip(),
                            "deviation": deviation_val,
                            "timestamp": now_str,
                            "date": str(fault_start_date)
                        }).eq("id", record_id).execute()
                        supabase.table("reason_audit_log").insert({
                            "action_type": "update",
                            "record_id": record_id,
                            "old_value": str(old_data),
                            "new_value": str({
                                "reason": reason_to_store,
                                "comment": comment.strip(),
                                "timestamp": now_str
                            }),
                            "timestamp": now_str
                        }).execute()
                        insert_count = 1
                    else:
                        # Insert
                        entry = {
                            "plant": plant_select[0],
                            "date": str(fault_start_date),
                            "input_name": selected_eq,
                            "deviation": deviation_val,
                            "reason": reason_to_store,
                            "comment": comment.strip(),
                            "timestamp": now_str,
                            "fault_start_date": str(fault_start_date),
                            "fault_end_date": str(fault_end_date)
                        }
                        resp = supabase.table("deviation_reasons").insert(entry).execute()
                        if resp.data:
                            insert_count = 1
                            supabase.table("reason_audit_log").insert({
                                "action_type": "insert",
                                "record_id": None,
                                "old_value": None,
                                "new_value": str(entry),
                                "timestamp": now_str
                            }).execute()
                    st.success(f"Reason/comment tagged for {selected_eq} from {fault_start_date} to {fault_end_date} ({insert_count} record).")
                    st.session_state["submit_success"] = True  # Set flag
                    st.rerun()  # Rerun to apply resets

        # ---- Reason/Comment Activity Matrix ---- (MIGRATED TO SUPABASE)
        st.markdown("---")
        st.markdown("#### 📅 Reason/Comment Activity Matrix")
        date_start_str = str(date_start)
        date_end_str = str(date_end)
        supabase = get_supabase_client()
        # Fetch overlapping faults
        matrix_query = supabase.table("deviation_reasons").select("*").eq("plant", plant_select[0]).lte("fault_start_date", date_end_str).gte("fault_end_date", date_start_str).execute()
        matrix_df = pd.DataFrame(matrix_query.data) if matrix_query.data else pd.DataFrame()

        if not matrix_df.empty:
            equipment_list = sorted(matrix_df['input_name'].unique())
            # Expand ranges for matrix [Requirement 4] Ensure legend
            expanded_rows = []
            for _, row in matrix_df.iterrows():
                start = pd.to_datetime(row['fault_start_date'])
                end = pd.to_datetime(row['fault_end_date'])
                for d in pd.date_range(start, end):
                    if date_start <= d.date() <= date_end:  # Filter to selected date range
                        new_row = row.copy()
                        new_row['plot_date'] = d.strftime("%Y-%m-%d")
                        expanded_rows.append(new_row)
            expanded_df = pd.DataFrame(expanded_rows)
            if expanded_df.empty:
                st.info("No tagged comments in the selected date range.")
            else:
                # Limit color map to present reasons to avoid extra legend entries
                present_reasons = expanded_df['reason'].unique()
                limited_color_map = {r: REASON_COLOR.get(r, "#888") for r in present_reasons}
                fig = px.scatter(
                    expanded_df,
                    x='plot_date',
                    y='input_name',
                    color='reason',
                    color_discrete_map=limited_color_map,  # Only present colors
                    hover_data={'comment': True}
                )
                fig.update_traces(marker=dict(size=22, line=dict(width=2, color="#333")))
                fig.update_layout(
                    title="Reason/Comment Tag Activity",
                    xaxis_title="Date",
                    yaxis_title="Equipment",
                    plot_bgcolor='white',
                    height=320 + 15 * len(equipment_list),
                    showlegend=True,  # [Requirement 4] Add legend
                    xaxis_tickformat='%Y-%m-%d'  # Ensure no time in ticks
                )
                st.plotly_chart(fig, use_container_width=True)
                # Remove dynamic legend HTML as per user request
        else:
            st.info("No tagged comments found for this plant and date range.")

        # ---- Editable Table Log Below ---- (MIGRATED TO SUPABASE)
        st.markdown("#### 📝 Recent Reasons/Comments Log")
        # [Requirement 2] Show Comments button
        if st.button("Show Comments", key="show_comments_equip"):
            if not matrix_df.empty:
                log_df = matrix_df.copy()
                log_df = log_df.sort_values(["fault_start_date", "timestamp"], ascending=[False, False])
                for idx, row in log_df.iterrows():
                    with st.expander(f"{row['fault_start_date']}→{row['fault_end_date']} | {row['input_name']} | {row['reason']}"):
                        st.markdown(f"**Comment:** {row['comment']}")
                        new_reason = st.selectbox("Edit Reason", REASON_LIST, index=REASON_LIST.index(row['reason']) if row['reason'] in REASON_LIST else len(REASON_LIST)-1, key=f"edit_reason_{idx}")
                        custom_edit_reason = ""
                        if new_reason == "Others":
                            custom_edit_reason = st.text_input("Custom Reason", row['reason'] if row['reason'] not in REASON_LIST else "", key=f"edit_custom_{idx}")
                        new_comment = st.text_area("Edit Comment", row['comment'], key=f"edit_comment_{idx}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Update", key=f"update_{idx}"):
                                if not new_comment.strip() or (new_reason == "Others" and not custom_edit_reason.strip()):
                                    st.error("Reason and Comment are mandatory.")
                                else:
                                    reason_final = custom_edit_reason.strip() if new_reason == "Others" else new_reason
                                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    supabase = get_supabase_client()
                                    match = supabase.table("deviation_reasons").select("*").eq("plant", row["plant"]).eq("fault_start_date", row["fault_start_date"]).eq("fault_end_date", row["fault_end_date"]).eq("input_name", row["input_name"]).eq("timestamp", row["timestamp"]).execute()
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
                                        st.success("Updated!")
                                        st.rerun()
                        with col2:
                            if st.button("Delete", key=f"delete_{idx}"):
                                supabase = get_supabase_client()
                                match = supabase.table("deviation_reasons").select("*").eq("plant", row["plant"]).eq("fault_start_date", row["fault_start_date"]).eq("fault_end_date", row["fault_end_date"]).eq("input_name", row["input_name"]).eq("timestamp", row["timestamp"]).execute()
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
                                    st.success("Deleted!")
                                    st.rerun()
                st.download_button(
                    "Download Reason/Comment Log (CSV)",
                    log_df[["fault_start_date", "fault_end_date", "input_name", "reason", "comment", "timestamp"]].to_csv(index=False),
                    "reason_log.csv",
                    "text/csv"
                )
            else:
                st.info("No log entries for this selection.")
        else:
            st.info("Click 'Show Comments' to load the filtered log.")

    # ----------- Root Cause Analytics Tab ----------- (MIGRATED TO SUPABASE)
    with tabs[2]:
        st.subheader("Root Cause Analytics & Remarks Log")
        supabase = get_supabase_client()
        # Fetch overlapping faults
        reason_query = supabase.table("deviation_reasons").select("*").in_("plant", plant_select).lte("fault_start_date", date_end_str).gte("fault_end_date", date_start_str).execute()
        reason_df = pd.DataFrame(reason_query.data) if reason_query.data else pd.DataFrame()
        if not reason_df.empty:
            # Change to average deviation
            reason_dev = reason_df.groupby('reason')['deviation'].mean().reset_index()
            reason_dev.columns = ['Reason', 'Avg Deviation (%)']
            pie = px.pie(
                reason_dev,
                names='Reason',
                values='Avg Deviation (%)',
                hole=0.4,
                color='Reason',
                color_discrete_map=REASON_COLOR  # [Requirement 3]
            )
            pie.update_layout(title="Distribution of Reasons for Underperformance (by Avg Deviation %)", showlegend=True)
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No reason/comment data for selection.")

        st.markdown("### Remarks Log (Edit/Delete)")
        # [Requirement 2] Show Comments button
        if st.button("Show Comments", key="show_comments_root"):
            if not reason_df.empty:
                for idx, row in reason_df.iterrows():
                    with st.expander(f"{row['fault_start_date']}→{row['fault_end_date']} | {row['input_name']} | {row['reason']}"):
                        st.write(f"Deviation: {row.get('deviation', 0.0):.2f}%")
                        new_reason = st.text_input("Edit Reason", row["reason"], key=f"reason_{idx}")
                        new_comment = st.text_area("Edit Comment", row["comment"], key=f"comment_{idx}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Update", key=f"update_root_{idx}"):
                                supabase = get_supabase_client()
                                match = supabase.table("deviation_reasons").select("*").eq("plant", row["plant"]).eq("fault_start_date", row["fault_start_date"]).eq("fault_end_date", row["fault_end_date"]).eq("input_name", row["input_name"]).execute()
                                if match.data:
                                    record_id = match.data[0]['id']
                                    old_data = match.data[0]
                                    supabase.table("deviation_reasons").update({
                                        "reason": new_reason,
                                        "comment": new_comment
                                    }).eq("id", record_id).execute()
                                    # Audit log
                                    supabase.table("reason_audit_log").insert({
                                        "action_type": "update",
                                        "record_id": record_id,
                                        "old_value": str(old_data),
                                        "new_value": str({
                                            "reason": new_reason,
                                            "comment": new_comment
                                        }),
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }).execute()
                                    st.success("Updated!")
                                    st.rerun()
                        with col2:
                            if st.button("Delete", key=f"delete_root_{idx}"):
                                supabase = get_supabase_client()
                                match = supabase.table("deviation_reasons").select("*").eq("plant", row["plant"]).eq("fault_start_date", row["fault_start_date"]).eq("fault_end_date", row["fault_end_date"]).eq("input_name", row["input_name"]).execute()
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
                                    st.success("Deleted!")
                                    st.rerun()
            else:
                st.info("No remarks to show.")

            if not reason_df.empty:
                st.download_button("Download Remarks Log (Excel)",
                                reason_df.to_csv(index=False), "remarks_log.csv", "text/csv")
        else:
            st.info("Click 'Show Comments' to load the filtered log.")