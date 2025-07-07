import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
import numpy as np

DB_PATH = "dgr_data.duckdb"
PERF_TABLE = "dgr_data"
REASON_TABLE = "deviation_reasons"

def render_visualisation_tab(plant_select, date_start, date_end, threshold):
    if not plant_select or not date_start or not date_end:
        st.info("Please select plants and date range to view visualisation.")
        return

    st.header("🌞 Visualisation & Root Cause Analytics")

    with duckdb.connect(DB_PATH) as con:
        perf_query = f"""
            SELECT * FROM {PERF_TABLE}
            WHERE plant IN ({','.join(['?']*len(plant_select))})
              AND date BETWEEN ? AND ?
        """
        perf_params = plant_select + [date_start, date_end]
        perf_df = con.execute(perf_query, perf_params).df()

        try:
            reason_query = f"""
                SELECT * FROM {REASON_TABLE}
                WHERE plant IN ({','.join(['?']*len(plant_select))})
                  AND date BETWEEN ? AND ?
            """
            reason_df = con.execute(reason_query, perf_params).df()
        except Exception:
            reason_df = pd.DataFrame(columns=["plant","date","input_name","value","reason","comment"])

    tabs = st.tabs(["Plant Overview", "Equipment Drilldown", "Root Cause Analytics"])

    # ---------- Plant Overview Tab ----------
    with tabs[0]:
        st.subheader("Plant Performance Comparison")
        if perf_df.empty:
            st.info("No performance data found for selection.")
        else:
            # --- Multi-plant (compare plants) ---
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
                # Color: red for negative, green for positive, intensity by value
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
            # --- Single plant, single date: show equipment-wise ---
            elif len(plant_select) == 1 and date_start == date_end:
                plant_name = plant_select[0]
                equip_df = perf_df[perf_df["plant"]==plant_name]
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
            # --- Single plant, date range: show daily trend ---
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

    # ----------- Equipment Drilldown Tab (unchanged) -----------
    with tabs[1]:
        if len(plant_select) != 1:
            st.info("Please select exactly one plant to view equipment drilldown.")
        else:
            st.subheader(f"Equipment Drilldown: {plant_select[0]}")
            plant_df = perf_df[perf_df["plant"] == plant_select[0]]
            if plant_df.empty:
                st.info("No equipment data for selected plant and range.")
            else:
                # --- Bar Chart: Equipment avg deviation, colored by sign ---
                equip_metrics = (
                    plant_df.groupby("input_name")["value"]
                    .mean()
                    .reset_index(name="Avg Deviation")
                )
                colors = ['red' if v < 0 else 'green' for v in equip_metrics["Avg Deviation"]]
                fig = go.Figure(go.Bar(
                    x=equip_metrics["input_name"],
                    y=equip_metrics["Avg Deviation"],
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in equip_metrics["Avg Deviation"]],
                    textposition='outside'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(
                    title=f"Avg Deviation by Equipment ({plant_select[0]})",
                    xaxis_title="Equipment", yaxis_title="Avg Deviation (%)",
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Trend for multiple selected equipment ---
                equipment_options = equip_metrics["input_name"].tolist()
                selected_equipment = st.multiselect(
                    "Show Trend for Equipment (select one or more to compare)",
                    equipment_options,
                    key="multi_trend_eq_select"
                )
                if selected_equipment:
                    trend_fig = go.Figure()
                    color_palette = [
                        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
                        "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
                        "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"
                    ]
                    for idx, eq in enumerate(selected_equipment):
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

                # --- Tag Reason/Comment for Equipment Underperformance (unchanged) ---
                st.markdown("**Tag Reason/Comment for Equipment Underperformance**")
                underperf_equip = plant_df[plant_df["value"] <= threshold]["input_name"].unique()
                selected_equip = st.multiselect("Select Equipment", underperf_equip.tolist(), key="multi_equip")
                reason_list = ["Soiling", "Inverter Fault", "Shadow", "Other"]
                reason = st.selectbox("Reason", reason_list, key="reason_box")
                custom_reason = ""
                if reason == "Other":
                    custom_reason = st.text_input("Custom Reason", key="custom_reason")
                comment = st.text_area("Comment (optional)", key="comment_box")
                if st.button("Add Reason/Comment", key="add_reason"):
                    with duckdb.connect(DB_PATH) as con:
                        for equip in selected_equip:
                            value = float(plant_df[(plant_df["input_name"] == equip)]["value"].iloc[-1])
                            con.execute(f"""
                                INSERT INTO {REASON_TABLE}
                                (plant, date, input_name, value, reason, comment)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (plant_select[0], date_end, equip, value,
                                  custom_reason if reason == "Other" else reason, comment))
                    st.success("Reason(s) logged!")

                # --- Pie chart of all reasons for this plant's equipment ---
                if not reason_df.empty:
                    plant_reason_df = reason_df[reason_df["plant"] == plant_select[0]]
                    if not plant_reason_df.empty:
                        reason_count = plant_reason_df["reason"].value_counts().reset_index()
                        reason_count.columns = ["Reason", "Count"]
                        pie = go.Figure(go.Pie(
                            labels=reason_count["Reason"], values=reason_count["Count"], hole=0.4
                        ))
                        pie.update_layout(title=f"Distribution of Reasons for {plant_select[0]}")
                        st.plotly_chart(pie, use_container_width=True)



    # ----------- Root Cause Analytics Tab (unchanged) -----------
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
                    st.write(f"Deviation: {row['value']}%")
                    new_reason = st.text_input("Edit Reason", row["reason"], key=f"reason_{idx}")
                    new_comment = st.text_area("Edit Comment", row["comment"], key=f"comment_{idx}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Update", key=f"update_{idx}"):
                            with duckdb.connect(DB_PATH) as con:
                                con.execute(f"""
                                    UPDATE {REASON_TABLE}
                                    SET reason=?, comment=?
                                    WHERE plant=? AND date=? AND input_name=?
                                """, (new_reason, new_comment, row["plant"], row["date"], row["input_name"]))
                            st.success("Updated!")
                    with col2:
                        if st.button("Delete", key=f"delete_{idx}"):
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
