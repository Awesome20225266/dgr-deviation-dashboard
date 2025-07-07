import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
from datetime import datetime

DB_PATH = "dgr_data.duckdb"
PERF_TABLE = "dgr_data"
REASON_TABLE = "deviation_reasons"

def render_visualisation_tab(plants_available, date_min, date_max):
    st.header("🌞 Visualisation & Root Cause Analytics")

    # --- Filter selection ---
    plant_select = st.multiselect("Select Plant(s)", plants_available, default=plants_available[:1])
    date_range = st.date_input(
        "Select Date Range", (date_min, date_max), min_value=date_min, max_value=date_max
    )
    threshold = st.number_input("Deviation Threshold (%)", value=-3.0, step=0.1)

    if st.button("Plot", key="viz_plot"):
        # --- Fetch Data ---
        with duckdb.connect(DB_PATH) as con:
            perf_query = f"""
                SELECT * FROM {PERF_TABLE}
                WHERE plant IN ({','.join(['?']*len(plant_select))})
                  AND date BETWEEN ? AND ?
            """
            perf_params = plant_select + [date_range[0], date_range[1]]
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

        # --- Tabs ---
        tabs = st.tabs(["Plant Overview", "Equipment Drilldown", "Root Cause Analytics"])

        # --- Plant Overview ---
        with tabs[0]:
            st.subheader("Plant-Wise Performance")
            if not perf_df.empty:
                plant_metrics = (
                    perf_df.groupby("plant")["value"]
                    .apply(lambda x: (threshold-x[x <= threshold]).sum())
                    .reset_index(name="Total Shortfall")
                )
                fig = go.Figure(go.Bar(
                    x=plant_metrics["plant"],
                    y=plant_metrics["Total Shortfall"],
                    marker_color="tomato"
                ))
                fig.update_layout(title="Total Shortfall by Plant", xaxis_title="Plant", yaxis_title="Total Shortfall")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data found for selection.")

        # --- Equipment Drilldown (only for one plant) ---
        if len(plant_select) == 1:
            with tabs[1]:
                st.subheader(f"Equipment Drilldown: {plant_select[0]}")
                plant_df = perf_df[perf_df["plant"] == plant_select[0]]
                if not plant_df.empty:
                    equip_metrics = (
                        plant_df.groupby("input_name")["value"]
                        .mean()
                        .reset_index(name="Avg Deviation")
                    )
                    fig = go.Figure(go.Bar(
                        x=equip_metrics["input_name"],
                        y=equip_metrics["Avg Deviation"],
                        marker_color="royalblue"
                    ))
                    fig.update_layout(title=f"Avg Deviation by Equipment ({plant_select[0]})",
                                      xaxis_title="Equipment", yaxis_title="Avg Deviation (%)")
                    st.plotly_chart(fig, use_container_width=True)
                    # --- Tag root cause for equipment ---
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
                                # For simplicity, use latest date in range
                                value = float(plant_df[(plant_df["input_name"] == equip)]["value"].iloc[0])
                                con.execute(f"""
                                    INSERT INTO {REASON_TABLE}
                                    (plant, date, input_name, value, reason, comment)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (plant_select[0], date_range[0], equip, value,
                                      custom_reason if reason == "Other" else reason, comment))
                        st.success("Reason(s) logged!")
                else:
                    st.info("No equipment data for selected plant and range.")
        else:
            with tabs[1]:
                st.info("Select exactly one plant for equipment drilldown.")

        # --- Root Cause Analytics ---
        with tabs[2]:
            st.subheader("Root Cause Analytics & Remarks Log")
            if not reason_df.empty:
                # Pie chart of reasons
                reason_count = reason_df["reason"].value_counts().reset_index()
                reason_count.columns = ["Reason", "Count"]
                pie = go.Figure(go.Pie(
                    labels=reason_count["Reason"], values=reason_count["Count"], hole=0.4
                ))
                pie.update_layout(title="Distribution of Reasons for Underperformance")
                st.plotly_chart(pie, use_container_width=True)
            else:
                st.info("No reason/comment data for selection.")

            # Editable remarks table
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

            # Download
            if not reason_df.empty:
                st.download_button("Download Remarks Log (Excel)",
                                   reason_df.to_csv(index=False), "remarks_log.csv", "text/csv")
