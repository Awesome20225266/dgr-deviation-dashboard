import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

DB_PATH = "dgr_data.duckdb"
PERF_TABLE = "dgr_data"
REASON_TABLE = "deviation_reasons"

REASON_LIST = [
    "Soiling", "Shadow", "Disconnected String", "Connector Burn", "Fuse Failure", "IGBT Failure", "Module Damage",
    "Power Clipping", "Vegetation Growth", "Bypass diode", "Degradation", "Temperature Loss", "RISO Fault",
    "MPPT Malfunction", "Grid Outage", "Load Curtailment", "Efficiency loss", "Ground Fault", "Module Mismatch",
    "IIGBT Issue", "Array Misalignment", "Tracker Failure", "Inverter Fan Issue", "Bifacial factor Loss",
    "Power Limitation", "Others"
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

def render_visualisation_tab(plant_select, date_start, date_end, threshold):
    global REASON_LIST

    # --- 1. Initialize session state (must be before widget definition!) ---
    for key, default in [
        ("reason_main_select", REASON_LIST[0]),
        ("custom_reason_box", ""),
        ("reason_comment_box", "")
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if not plant_select or not date_start or not date_end:
        st.info("Please select plants and date range to view visualisation.")
        return

    st.header("🌞 Visualisation & Root Cause Analytics")

    # Ensure reasons table exists (safe to do every run)
    with duckdb.connect(DB_PATH) as con:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {REASON_TABLE} (
                plant TEXT,
                date TEXT,
                input_name TEXT,
                reason TEXT,
                comment TEXT,
                timestamp TEXT
            );
        """)

    with duckdb.connect(DB_PATH) as con:
        perf_query = f"""
            SELECT * FROM {PERF_TABLE}
            WHERE plant IN ({','.join(['?']*len(plant_select))})
              AND date BETWEEN ? AND ?
        """
        perf_params = plant_select + [str(date_start), str(date_end)]
        perf_df = con.execute(perf_query, perf_params).df()
        try:
            reason_query = f"""
                SELECT * FROM {REASON_TABLE}
                WHERE plant IN ({','.join(['?']*len(plant_select))})
                  AND date BETWEEN ? AND ?
            """
            reason_df = con.execute(reason_query, perf_params).df()
        except Exception:
            reason_df = pd.DataFrame(columns=["plant", "date", "input_name", "value", "reason", "comment"])

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

    # ----------- Equipment Drilldown Tab -----------
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
            f"<span style='color:green'>Green</span>: ≥ 0% &nbsp;&nbsp;"
            f"<span style='color:orange'>Orange</span>: 0% to {threshold_value}% &nbsp;&nbsp;"
            f"<span style='color:red'>Red</span>: &lt; {threshold_value}%",
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

        # ------- Tag Reason/Comment Section -------
        st.markdown("---")
        st.markdown("### 🏷️ Tag Reason/Comment for Equipment Underperformance")
        eq_dropdown_options = [f"🌐 Whole Plant"] + list(equip_metrics['emoji_label'])
        selected_label = st.selectbox("Select Equipment (with status badge)", eq_dropdown_options, key="reason_equip_select")
        if selected_label.startswith("🌐"):
            selected_eq = "Whole Plant"
        else:
            selected_eq = selected_label.split(' ', 1)[1].split(' (')[0]
        reason = st.selectbox("Select Reason*", REASON_LIST, key="reason_main_select")
        if reason == "Others":
            custom_reason = st.text_input("Custom Reason*", key="custom_reason_box")
        else:
            custom_reason = ""
        comment = st.text_area("Comment (Action/Status)*", placeholder="E.g. Issue resolved, cleaning scheduled, replacement pending...", key="reason_comment_box")
        submit_disabled = (
            (not selected_eq)
            or (reason == "Others" and not custom_reason.strip())
            or (reason != "Others" and not reason)
            or (not comment.strip())
        )
        if st.button("Submit Reason/Comment", disabled=submit_disabled):
            with duckdb.connect(DB_PATH) as con:
                con.execute(
                    f"""INSERT INTO {REASON_TABLE}
                    (plant, date, input_name, reason, comment, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        plant_select[0],
                        str(date_end),
                        selected_eq,
                        custom_reason.strip() if reason == "Others" else reason,
                        comment.strip(),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
            st.success(f"Reason/comment tagged for {selected_eq} on {date_end}.")
            st.rerun()

        # ---- Reason/Comment Activity Matrix ----
        st.markdown("#### 📅 Reason/Comment Activity Matrix")
        date_start_str = str(date_start)
        date_end_str = str(date_end)
        with duckdb.connect(DB_PATH) as con:
            query = f"""
                SELECT plant, date, input_name, reason, comment, timestamp
                FROM {REASON_TABLE}
                WHERE plant = ? 
                  AND CAST(date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            """
            matrix_df = con.execute(query, (plant_select[0], date_start_str, date_end_str)).df()

        if not matrix_df.empty:
            equipment_list = sorted(matrix_df['input_name'].unique())
            date_list = sorted(matrix_df['date'].unique())
            reason_map = {r: REASON_COLOR.get(r, "#888") for r in matrix_df['reason'].unique()}
            scatter_x, scatter_y, scatter_color, scatter_text = [], [], [], []
            for idx, equip in enumerate(equipment_list):
                for jdx, d in enumerate(date_list):
                    sel = matrix_df[(matrix_df['input_name'] == equip) & (matrix_df['date'] == d)]
                    if not sel.empty:
                        row = sel.iloc[-1]  # Latest only
                        scatter_x.append(d)
                        scatter_y.append(equip)
                        scatter_color.append(REASON_COLOR.get(row['reason'], "#888"))
                        scatter_text.append(
                            f"Date: {d}<br>Equipment: {equip}<br>Reason: {row['reason']}<br>Comment: {row['comment']}"
                        )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scatter_x,
                y=scatter_y,
                mode="markers",
                marker=dict(
                    color=scatter_color,
                    size=22,
                    line=dict(width=2, color="#333"),
                    symbol='circle'
                ),
                text=scatter_text,
                hoverinfo="text"
            ))
            fig.update_xaxes(
                title="Date",
                type="category",
                showgrid=True,
                gridwidth=1, gridcolor='#ccc',
                tickangle=45
            )
            fig.update_yaxes(
                title="Equipment",
                type="category",
                showgrid=True,
                gridwidth=1, gridcolor='#ccc'
            )
            legend_items = []
            for k, v in reason_map.items():
                legend_items.append(f"<span style='color:{v}; font-weight:bold;'>&#11044;</span> {k}")
            st.markdown(
                "**Legend:**<br>" + " &nbsp; ".join(legend_items),
                unsafe_allow_html=True
            )
            fig.update_layout(
                title="Reason/Comment Tag Activity",
                plot_bgcolor='white',
                margin=dict(l=40, r=20, t=45, b=45),
                height=320 + 15 * len(equipment_list)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tagged comments found for this plant and date range.")

        # ---- Editable Table Log Below ----
        st.markdown("#### 📝 Recent Reasons/Comments Log")
        if not matrix_df.empty:
            log_df = matrix_df.copy()
            log_df = log_df.sort_values(["date", "timestamp"], ascending=[False, False])
            for idx, row in log_df.iterrows():
                with st.expander(f"{row['date']} | {row['input_name']} | {row['reason']}"):
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
                                with duckdb.connect(DB_PATH) as con:
                                    con.execute(f"""
                                        UPDATE {REASON_TABLE}
                                        SET reason=?, comment=?
                                        WHERE plant=? AND date=? AND input_name=? AND timestamp=?
                                    """, (
                                        custom_edit_reason.strip() if new_reason == "Others" else new_reason,
                                        new_comment.strip(),
                                        row['plant'], row['date'], row['input_name'], row['timestamp']
                                    ))
                                st.success("Updated!")
                                st.rerun()
                    with col2:
                        if st.button("Delete", key=f"delete_{idx}"):
                            with duckdb.connect(DB_PATH) as con:
                                con.execute(f"""
                                    DELETE FROM {REASON_TABLE}
                                    WHERE plant=? AND date=? AND input_name=? AND timestamp=?
                                """, (row['plant'], row['date'], row['input_name'], row['timestamp']))
                            st.success("Deleted!")
                            st.rerun()
            st.download_button(
                "Download Reason/Comment Log (CSV)",
                log_df[["date", "input_name", "reason", "comment", "timestamp"]].to_csv(index=False),
                "reason_log.csv",
                "text/csv"
            )
        else:
            st.info("No log entries for this selection.")

    # ----------- Root Cause Analytics Tab -----------
    with tabs[2]:
        st.subheader("Root Cause Analytics & Remarks Log")
        if not reason_df.empty:
            reason_count = reason_df["reason"].value_counts().reset_index()
            reason_count.columns = ["Reason", "Count"]
            pie = go.Figure(go.Pie(
                labels=reason_count["Reason"], values=reason_count["Count"], hole=0.4
            ))
            pie.update_layout(title="Distribution of Reasons for Underperformance")
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No reason/comment data for selection.")

        st.markdown("### Remarks Log (Edit/Delete)")
        if not reason_df.empty:
            for idx, row in reason_df.iterrows():
                with st.expander(f"{row['date']} | {row['input_name']} | {row['reason']}"):
                    st.write(f"Deviation: {row.get('value', '')}%")
                    new_reason = st.text_input("Edit Reason", row["reason"], key=f"reason_{idx}")
                    new_comment = st.text_area("Edit Comment", row["comment"], key=f"comment_{idx}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Update", key=f"update_root_{idx}"):
                            with duckdb.connect(DB_PATH) as con:
                                con.execute(f"""
                                    UPDATE {REASON_TABLE}
                                    SET reason=?, comment=?
                                    WHERE plant=? AND date=? AND input_name=?
                                """, (new_reason, new_comment, row["plant"], row["date"], row["input_name"]))
                            st.success("Updated!")
                    with col2:
                        if st.button("Delete", key=f"delete_root_{idx}"):
                            with duckdb.connect(DB_PATH) as con:
                                con.execute(f"""
                                    DELETE FROM {REASON_TABLE}
                                    WHERE plant=? AND date=? AND input_name=?
                                """, (row["plant"], row["date"], row["input_name"]))
                            st.success("Deleted!")
        else:
            st.info("No remarks to show.")

        if not reason_df.empty:
            st.download_button("Download Remarks Log (Excel)",
                               reason_df.to_csv(index=False), "remarks_log.csv", "text/csv")
