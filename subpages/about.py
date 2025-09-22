#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/21 14:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("- Built with Python, Streamlit, and Keras")
    caption("- Data preprocessing includes handling missing values, scaling, and encoding")
    caption("- Training module provides batch size and epochs configuration")
    caption("- Visualization module displays scatter plots, tables, ROC curves, and decision boundaries")
    caption("- Suitable for educational demos and beginner ML experiments")
