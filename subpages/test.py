#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/22 17:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   test.py
# @Desc     :   

from numpy import sqrt
from os import path
from pandas import DataFrame
from plotly.express import line
from sklearn.metrics import accuracy_score, r2_score, roc_curve, auc, mean_squared_error, mean_absolute_error
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, columns, metric)
from subpages.train import MODEL_PATH
from tensorflow.keras.models import load_model

from utils.helper import Timer, scatter_visualiser, decision_boundary_adder

empty_messages: empty = empty()
empty_result_title: empty = empty()
acc_col, r2_col, mse_col, rMse_col, mae_col = columns(5, gap="small")
empty_roc_title: empty = empty()
empty_roc_chart: empty = empty()
empty_test_title: empty = empty()
empty_test_chart: empty = empty()
empty_test_table: empty = empty()

pre_sessions: list[str] = ["raw"]
for session in pre_sessions:
    session_state.setdefault(session, None)
split_sessions: list[str] = ["X_Train", "X_Test", "Y_Train", "Y_Test"]
for session in split_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["vTimer", "Y_Pred", "mlp"]
for session in test_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["raw"] is None:
        empty_messages.error("Please upload the dataset in the Home page first.")
    else:
        if session_state["X_Train"] is None:
            empty_messages.error("Please split the dataset in the Data Preparation page first.")
        else:
            if not path.exists(MODEL_PATH):
                empty_messages.warning("Please train the model in the Model Training page first.")
            else:
                subheader("Model Testing Settings")

                empty_test_title.markdown(f"#### Testing dataset {session_state["X_Test"].shape}")
                empty_test_table.data_editor(session_state["X_Test"], hide_index=True, disabled=True, width="stretch")
                empty_test_chart.scatter_chart(
                    session_state["raw"],
                    x=session_state["raw"].columns[0],
                    y=session_state["raw"].columns[1],
                    color=session_state["raw"].columns[-1],
                    use_container_width=True
                )

                if session_state["Y_Pred"] is None:
                    empty_messages.info(
                        "The testing dataset and trained model are ready. You can start testing the model."
                    )

                    if button("Start Model Testing", type="primary", width="stretch"):
                        with spinner("Testing the model...", show_time=True, width="stretch"):
                            with Timer("Testing the model") as session_state["vTimer"]:
                                session_state["mlp"] = load_model(MODEL_PATH)
                                session_state["Y_Pred"] = session_state["mlp"].predict(session_state["X_Test"])
                        rerun()
                else:
                    empty_messages.success(
                        f"{session_state["vTimer"]} The model has been tested. You can view the prediction results below."
                    )

                    y_true = session_state["Y_Test"].astype(float)
                    y_pred = session_state["Y_Pred"].round()
                    empty_result_title.markdown("#### Model Performance Metrics")
                    with acc_col:
                        accuracy = accuracy_score(y_true, y_pred)
                        metric("Accuracy", f"{accuracy:.2%}", delta=None, delta_color="normal")
                    with r2_col:
                        r2value = r2_score(y_true, y_pred)
                        metric("R² Score", f"{r2value:.4f}", delta=None, delta_color="normal")
                    with mse_col:
                        mse = mean_squared_error(y_true, y_pred)
                        metric("MSE", f"{mse:.4f}", delta=None, delta_color="normal")
                    with rMse_col:
                        rMse = sqrt(mse)
                        metric("rMSE", f"{rMse:.4f}", delta=None, delta_color="normal")
                    with mae_col:
                        mae = mean_absolute_error(y_true, y_pred)
                        metric("MAE", f"{mae:.4f}", delta=None, delta_color="normal")

                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    roc_auc = auc(fpr, tpr)
                    fig = line(x=fpr, y=tpr, labels={"x": "FPR", "y": "TPR"})
                    empty_roc_title.markdown(f"#### ROC Curve (AUC = {roc_auc:.4f})")
                    empty_roc_chart.plotly_chart(fig, use_container_width=True)

                    x = session_state["raw"].iloc[:, :-1]
                    y = DataFrame(session_state["raw"].iloc[:, -1], columns=[session_state["raw"].columns[-1]])
                    y_test: DataFrame = DataFrame(session_state["Y_Pred"], columns=["Prediction"])
                    fig = scatter_visualiser(x, y)
                    fig = decision_boundary_adder(fig, session_state["mlp"], session_state["X_Test"])
                    empty_test_chart.plotly_chart(fig, theme="streamlit", use_container_width=True)

                    if button("Retest the Model", type="secondary", width="stretch"):
                        for session in test_sessions:
                            session_state[session] = None
                        rerun()
