#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/21 17:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :   

from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, slider, number_input, caption)

from utils.helper import Timer

empty_messages: empty = empty()
empty_train_title: empty = empty()
empty_train_chart: empty = empty()
empty_train_table: empty = empty()
empty_test_title: empty = empty()
empty_test_chart: empty = empty()
empty_test_table: empty = empty()

pre_sessions: list[str] = ["raw", "hTimer"]
for session in pre_sessions:
    session_state.setdefault(session, None)
split_sessions: list[str] = ["sTimer", "X_Train", "X_Test", "Y_Train", "Y_Test"]
for session in split_sessions:
    session_state.setdefault(session, None)
model_sessions: list[str] = ["model", "mTimer"]
for session in model_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["raw"] is None:
        empty_messages.error("Please upload the dataset in the Home page first.")
    else:
        subheader("Model Training Settings")

        if session_state["X_Train"] is None:
            empty_messages.warning(
                "The data has been loaded successfully! Please split the dataset into training and testing sets."
            )

            test_size: float = slider(
                "Test Size (as a fraction of the dataset)",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.01,
                help="The proportion of the dataset to include in the test split.",
            )
            caption(f"Current Test Size is **{test_size:.1%}** of the dataset.")
            random_state: int = number_input(
                "Random State (for reproducibility)",
                min_value=0,
                max_value=100,
                value=27,
                step=1,
                help="Controls the shuffling applied to the data before applying the split.",
            )

            if button("Split the Dataset", type="primary", width="stretch"):
                with spinner("Splitting the dataset...", show_time=True, width="stretch"):
                    with Timer("Splitting the dataset") as session_state["sTimer"]:
                        cols: list[str] = session_state["raw"].columns.tolist()
                        X = session_state["raw"][cols[:-1]]
                        Y = session_state["raw"][cols[-1]]
                        print(type(X), type(Y))

                        (
                            session_state["X_Train"],
                            session_state["X_Test"],
                            session_state["Y_Train"],
                            session_state["Y_Test"]
                        ) = train_test_split(
                            X,
                            Y,
                            test_size=test_size,
                            random_state=random_state,
                            shuffle=True,
                        )
                rerun()
        else:
            train = concat([session_state["X_Train"], session_state["Y_Train"]], axis=1)
            print(type(train))
            empty_train_title.markdown(f"#### Training Set {session_state["X_Train"].shape}")
            empty_train_table.data_editor(train, hide_index=True, disabled=True, width="stretch")
            empty_train_chart.scatter_chart(
                train, x=train.columns[0], y=train.columns[1], color=train.columns[-1], use_container_width=True
            )

            test = concat([session_state["X_Test"], session_state["Y_Test"]], axis=1)
            print(type(test))
            empty_test_title.markdown(f"#### Test Set {session_state["X_Test"].shape}")
            empty_test_table.data_editor(test, hide_index=True, disabled=True, width="stretch")
            empty_test_chart.scatter_chart(
                test, x=test.columns[0], y=test.columns[1], color=test.columns[-1], use_container_width=True
            )

            if button("Clear the Split Dataset", type="secondary", width="stretch"):
                for session in split_sessions:
                    session_state[session] = None
                rerun()

            if session_state["model"] is None:
                empty_messages.info(
                    f"{session_state["sTimer"]} The dataset has been split successfully. You can proceed with model training."
                )

                if button("Train the MLP Model", type="primary", width="stretch"):
                    with spinner("Training the mlp model...", show_time=True, width="stretch"):
                        with Timer("Training the mlp model") as session_state["mTimer"]:
                            input_dims: int = session_state["X_Train"].shape[1]
                            # Set the Sequential model
                            session_state["model"] = Sequential(
                                Dense(32, input_dim=input_dims, activation="sigmoid"),
                                Dense(1, activation="sigmoid")
                            )
                            # Check the model summary
                            print(session_state["model"].summary())

                            session_state["model"].compile(
                                loss="binary_crossentropy",
                                optimizer="adam",
                                metrics=["accuracy", "Precision", "Recall", "AUC"]
                            )
                            pass

            else:
                empty_messages.success(f"{session_state["mTimer"]} The model has been trained successfully!")
