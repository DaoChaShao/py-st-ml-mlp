<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This project focuses on building and training a simple **Multi-Layer Perceptron (MLP)** model for binary classification
using the **Exclusive XOR Playground** dataset. The dataset is a classic example of a
**non-linearly separable problem**, making it an ideal playground for experimenting with neural networks and
understanding how MLPs can learn complex decision boundaries.

This project provides a complete machine learning workflow demo using Streamlit and Keras. It covers data preprocessing,
model training, and testing. Users can upload datasets, handle missing or duplicate values, visualize data
distributions, and train a Multi-Layer Perceptron (MLP) model for binary classification.

**DATA DESCRIPTION**
---
Dataset Name: [Exclusive XOR Playground](https://www.kaggle.com/datasets/martininf1n1ty/exclusive-xor-dataset)

+ **Number of Features**: 2 (x1, x2)
+ **Number of Classes**: 2 (binary classification)
+ **Data Pattern**:
    - The data points are distributed in a XOR pattern:
        - Top-left and bottom-right belong to Class 0
        - Top-right and bottom-left belong to Class 1
+ **Number of Samples**: [Insert number of samples if known, e.g., 200]
+ **Usage**: The dataset is mainly used for testing non-linear classifiers such as neural networks, decision trees, or
  kernel-based SVMs.
+ **Notes**: Traditional linear models cannot solve this problem effectively due to the XOR pattern, making it a great
  example for learning the power of hidden layers in MLPs.

**FEATURES**
---

- **Data Loading & Visualization**: Upload CSV datasets and view them in tables and scatter plots.
- **Data Cleaning**: Automatically detect missing values and duplicate rows, with options to clean them.
- **Data Splitting**: Split datasets into training and testing sets with customizable test size and random seed.
- **Model Training**: Build and train an MLP model with configurable batch size and epochs.
- **Training Monitoring**: Track loss, accuracy, precision, recall, and AUC metrics in real-time.
- **Model Testing**: Predict using the trained model and display accuracy, R², MSE, MAE, and ROC curve.
- **Decision Boundary Visualization**: Dynamically show the decision boundary on scatter plots.

**WEB DEVELOPMENT**
---

1. Install NiceGUI with the command `pip install streamlit`.
2. Run the command `pip show streamlit` or `pip show streamlit | grep Version` to check whether the package has been
   installed and its version.
3. Run the command `streamlit run app.py` to start the web application.

**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause License](LICENSE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command `pip install git-changelog`.
2. Run the command `pip show git-changelog` or `pip show git-changelog | grep Version` to check whether the changelog
   package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command `git-changelog`, creating the `Changelog.md` file.
6. Add the file `Changelog.md` to version control with the command `git add Changelog.md` or using the UI interface.
7. Run the command `git-changelog --output CHANGELOG.md` committing the changes and updating the changelog.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.