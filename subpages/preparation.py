#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/21 15:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :

from pandas import DataFrame, read_csv
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun)

from utils.helper import Timer, scatter_visualiser

empty_messages: empty = empty()
empty_raw_title: empty = empty()
empty_raw_chart: empty = empty()
empty_raw_table: empty = empty()
empty_plot_title: empty = empty()
empty_plot_chart: empty = empty()

home_sessions: list[str] = ["raw", "hTimer"]
for session in home_sessions:
    session_state.setdefault(session, None)

DATAPATH: str = "data/xor.csv"

with sidebar:
    subheader("Data Preparation Settings")

    if session_state["raw"] is None:
        empty_messages.error("Please upload the dataset in the Home page first.")

        if button("Load the Raw Dataset", type="primary", width="stretch"):
            with spinner("Loading the raw dataset...", show_time=True, width="stretch"):
                with Timer("Loading the raw dataset") as t:
                    session_state["raw"] = read_csv(DATAPATH)
                    # print(type(session_state["raw"]))
                    session_state["raw"].drop(session_state["raw"].columns[0], axis=1, inplace=True)
                    session_state["hTimer"] = t
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

        if button("Clear the Raw Dataset", type="secondary", width="stretch"):
            for session in home_sessions:
                session_state[session] = None
            rerun()
