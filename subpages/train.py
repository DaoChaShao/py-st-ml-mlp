#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/21 17:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :

from os import path, remove
from keras.models import Sequential
from keras.layers import Dense, Input
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, slider, number_input, caption,
                       columns, metric)
from tensorflow.keras import metrics

from utils.helper import Timer, TFKerasLogger

empty_messages: empty = empty()

empty_result_title: empty = empty()
loss_col, acc_col, pre_col, rec_col, auc_col = columns(5, gap="small")
loss_valid_col, acc_valid_col, pre_valid_col, rec_valid_col, auc_valid_col = columns(5, gap="small")
placeholder_loss = loss_col.empty()
placeholder_acc = acc_col.empty()
placeholder_pre = pre_col.empty()
placeholder_rec = rec_col.empty()
placeholder_auc = auc_col.empty()
placeholder_loss_val = loss_valid_col.empty()
placeholder_acc_val = acc_valid_col.empty()
placeholder_pre_val = pre_valid_col.empty()
placeholder_rec_val = rec_valid_col.empty()
placeholder_auc_val = auc_valid_col.empty()
empty_result_chart: empty = empty()

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
model_sessions: list[str] = ["model", "mTimer", "histories"]
for session in model_sessions:
    session_state.setdefault(session, None)

MODEL_PATH: str = "mlp_model.h5"

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

            # Initialize the metrics placeholders
            placeholders: dict = {
                "loss": placeholder_loss,
                "accuracy": placeholder_acc,
                "precision": placeholder_pre,
                "recall": placeholder_rec,
                "auc": placeholder_auc,
                "val_loss": placeholder_loss_val,
                "val_accuracy": placeholder_acc_val,
                "val_precision": placeholder_pre_val,
                "val_recall": placeholder_rec_val,
                "val_auc": placeholder_auc_val
            }
            # Convert DataFrame to numpy array for model training
            X_Train_array = session_state["X_Train"].to_numpy(dtype="float32")
            Y_Train_array = session_state["Y_Train"].to_numpy(dtype="float32")
            X_Test_array = session_state["X_Test"].to_numpy(dtype="float32")
            Y_Test_array = session_state["Y_Test"].to_numpy(dtype="float32")
            # Initialise the callback for visualisation
            callback = TFKerasLogger(placeholders)
            if session_state["model"] is None:
                empty_messages.info(
                    f"{session_state["sTimer"]} The dataset has been split successfully. You can proceed with model training."
                )

                batch_size: int = number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=256,
                    value=32,
                    step=1,
                    help="Number of samples per gradient update. Default is 32.",
                )

                epochs: int = number_input(
                    "Epochs",
                    min_value=1,
                    max_value=5000,
                    value=1000,
                    step=1,
                    help="Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.",
                )
                caption("**32** batch size and **1,000** epochs will be recommended for training.")

                if button("Train the MLP Model", type="primary", width="stretch"):
                    with spinner("Training the mlp model...", show_time=True, width="stretch"):
                        with Timer("Training the mlp model") as session_state["mTimer"]:
                            # Set the Sequential model
                            session_state["model"] = Sequential([
                                Input(shape=(X_Train_array.shape[1],)),
                                Dense(32, activation="sigmoid"),
                                Dense(1, activation="sigmoid")
                            ])

                            # Check the model summary
                            print(session_state["model"].summary())

                            session_state["model"].compile(
                                loss="binary_crossentropy",
                                optimizer="adam",
                                metrics=[
                                    "accuracy",
                                    metrics.Precision(name="precision"),
                                    metrics.Recall(name="recall"),
                                    metrics.AUC(name="auc"),
                                ],
                            )

                            # Train the model
                            session_state["model"].fit(
                                X_Train_array, Y_Train_array,
                                validation_data=(X_Train_array, Y_Train_array),
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0,
                                # Run the callback at the end of each epoch
                                callbacks=[callback]
                            )

                            # Store the training history
                            session_state["histories"] = callback.get_history()
                    rerun()
            else:
                empty_result_title.markdown("#### Training Results")
                hist = session_state["histories"]
                if hist:
                    last_epoch = len(hist["loss"])
                    for key, placeholder in placeholders.items():
                        if key in hist and placeholder is not None:
                            value = hist[key][-1]
                            label = f"Epoch {last_epoch}: {key.replace("val_", "Val ").capitalize()}"
                            placeholder.metric(label=label, value=f"{value:.4f}")

                if path.exists(MODEL_PATH):
                    empty_messages.info(
                        f"The model file **{MODEL_PATH}** already exists in the current directory.")

                    if button("Delete the Model", type="secondary", width="stretch"):
                        with spinner("Deleting the model...", show_time=True, width="stretch"):
                            with Timer("Deleting the model") as timer:
                                remove(MODEL_PATH)

                                for placeholder in placeholders.values():
                                    placeholder.empty()

                                for session in model_sessions:
                                    session_state[session] = None
                        empty_messages.success(f"{timer} The model has been deleted successfully!")
                        rerun()
                else:
                    empty_messages.success(
                        f"{session_state["mTimer"]} The model has been trained successfully! You can save the model."
                    )

                    if button("Save the Model", type="primary", width="stretch"):
                        with spinner("Saving the model...", show_time=True, width="stretch"):
                            with Timer("Saving the model") as timer:
                                session_state["model"].save(MODEL_PATH)
                        empty_messages.success(f"{timer} The model has been saved successfully!")
                        rerun()

            if button("Clear the Split Dataset", type="secondary", width="stretch"):
                for session in split_sessions:
                    session_state[session] = None
                rerun()
