#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/21 15:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :

from pandas import DataFrame, read_csv
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, columns, metric)

from utils.helper import Timer, scatter_visualiser

empty_messages: empty = empty()
nan_col, dup_col = columns(2, gap="small")
empty_raw_title: empty = empty()
empty_raw_chart: empty = empty()
empty_raw_table: empty = empty()
empty_plot_title: empty = empty()
empty_plot_chart: empty = empty()

pre_sessions: list[str] = ["raw", "hTimer"]
for session in pre_sessions:
    session_state.setdefault(session, None)

DATAPATH: str = "data/xor.csv"

with sidebar:
    subheader("Data Preparation Settings")

    if session_state["raw"] is None:
        empty_messages.error("Please upload the dataset in the Home page first.")

        if button("Load the Raw Dataset", type="primary", width="stretch"):
            with spinner("Loading the raw dataset...", show_time=True, width="stretch"):
                with Timer("Loading the raw dataset") as session_state["hTimer"]:
                    session_state["raw"] = read_csv(DATAPATH)
                    # print(type(session_state["raw"]))
                    session_state["raw"].drop(session_state["raw"].columns[0], axis=1, inplace=True)
            rerun()
    else:
        empty_messages.success(
            f"{session_state["hTimer"]} The raw dataset has been loaded. You can proceed with data preparation."
        )

        cols: list[str] = session_state["raw"].columns.tolist()
        # Transfer the last column to string for scatter chart colouring purpose
        session_state["raw"][cols[-1]] = session_state["raw"][cols[-1]].astype(str)

        empty_raw_title.markdown(f"#### Raw Dataset {session_state["raw"].shape}")
        empty_raw_table.data_editor(session_state["raw"], hide_index=True, disabled=True, width="stretch")
        empty_raw_chart.scatter_chart(
            session_state["raw"],
            x=cols[0],
            y=cols[1],
            color=cols[-1],
            use_container_width=True
        )

        data = session_state["raw"].iloc[:, :-1]
        # Use the values of y because y has been transferred string at previous session
        categories = DataFrame(session_state["raw"].iloc[:, -1].values, columns=["Category"])
        empty_plot_title.markdown("#### Plotly Scatter Plot of the Raw Dataset")
        fig = scatter_visualiser(data, categories, 1)
        empty_plot_chart.plotly_chart(fig, use_container_width=True)

        nan = session_state["raw"].isna().sum().sum()
        dup = session_state["raw"].duplicated().sum()
        with nan_col:
            metric("Missing Values", nan, delta=None, delta_color="inverse")
        with dup_col:
            metric("Duplicate Rows", dup, delta=None, delta_color="off")

        if nan > 0:
            if button("Drop Missing Values", type="primary", width="stretch"):
                with spinner("Dropping missing values...", show_time=True, width="stretch"):
                    with Timer("Dropping missing values") as timer:
                        session_state["raw"].dropna(inplace=True)
                    empty_messages.success(f"{timer} Missing values have been dropped.")
                rerun()
        if dup > 0:
            if button("Drop Duplicate Rows", type="primary", width="stretch"):
                with spinner("Dropping duplicate rows...", show_time=True, width="stretch"):
                    with Timer("Dropping duplicate rows") as timer:
                        session_state["raw"].drop_duplicates(inplace=True)
                    empty_messages.success(f"{timer} Duplicate rows have been dropped.")
                rerun()

        if button("Clear the Raw Dataset", type="secondary", width="stretch"):
            for session in pre_sessions:
                session_state[session] = None
            rerun()
