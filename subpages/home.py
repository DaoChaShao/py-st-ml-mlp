#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/21 14:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("ML - Make Moons")
with expander("**INTRODUCTION**", expanded=True):
    caption("+ Machine Learning Workflow Demo using Streamlit and Keras")
    caption("+ Supports data upload, visualization, and cleaning operations")
    caption("+ Interactive interface to train MLP models and monitor metrics in real-time")
    caption("+ Supports model testing and decision boundary visualization")
