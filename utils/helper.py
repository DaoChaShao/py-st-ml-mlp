#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/21 14:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :

from io import StringIO
from numpy import meshgrid, linspace, c_
from pandas import DataFrame, concat
from plotly.express import scatter, scatter_3d
from plotly.graph_objects import Contour
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from time import perf_counter


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("-" * 50)
        print(f"{self._description} has started.")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            # print("-" * 50)
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


def scatter_visualiser(data: DataFrame, categories: DataFrame = None, dims: int = 1):
    """ Visualise the data using scatter plots.
    :param data: the DataFrame containing the data
    :param categories: the DataFrame containing the categories for colouring and symbolising the data points
    :param dims: number of dimensions to reduce to if data has more than 3 dimensions (2 or 3)
    :return: a scatter plot with different colours and symbols for each category
    """
    if categories is not None:
        df = concat([data, categories], axis=1)
        category_name = categories.columns[0]
    else:
        df = data
        category_name = None
        print(category_name)

    fig = None

    match dims:
        case 1:
            dimensions = data.shape[1]
            if dimensions == 2:
                fig = scatter(
                    df,
                    x=data.columns[0],
                    y=data.columns[1],
                    color=category_name,
                    symbol=category_name,
                    hover_data=[data.columns[0], data.columns[1], category_name]
                ).update_layout(coloraxis_showscale=False)
            else:
                fig = scatter_3d(
                    df,
                    x=data.columns[0],
                    y=data.columns[1],
                    z=data.columns[2],
                    color=category_name,
                    symbol=category_name,
                    hover_data=[data.columns[0], data.columns[1], data.columns[2], category_name]
                ).update_layout(coloraxis_showscale=False)
        case 2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(data)
            df = DataFrame(components, columns=["PAC-X", "PAC-Y"])
            fig = scatter(
                df,
                x="PAC-X",
                y="PAC-Y",
                color=category_name,
                symbol=category_name,
                hover_data=["PAC-X", "PAC-Y"] + ([category_name] if category_name else [])
            ).update_layout(coloraxis_showscale=False)
        case 3:
            pca = PCA(n_components=3)
            components = pca.fit_transform(data)
            df = DataFrame(components, columns=["PAC-X", "PAC-Y", "PAC-Z"])
            fig = scatter_3d(
                df,
                x="PAC-X",
                y="PAC-Y",
                z="PAC-Z",
                color=category_name,
                symbol=category_name,
                hover_data=["PAC-X", "PAC-Y", "PAC-Z"] + ([category_name] if category_name else [])
            ).update_layout(coloraxis_showscale=False)
        case _:
            raise ValueError("dims must be 1, 2, or 3")

    return fig


def data_preprocessor(selected_data: DataFrame) -> tuple[DataFrame, StandardScaler]:
    """ Preprocess the data by handling missing values, scaling numerical features, and encoding categorical features.
    :param selected_data: the DataFrame containing the selected features for training
    :return: a DataFrame containing the preprocessed features
    """
    cols_num = selected_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_type = selected_data.select_dtypes(include=["object", "category"]).columns.tolist()
    # print(f"The cols filed with number: {cols_num}")
    # print(f"The cols filled with category: {cols_type}")

    # Establish a pipe to process numerical features
    pipe_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Set a list of transformers for the ColumnTransformer
    transformers = [("number", pipe_num, cols_num)]

    # Establish a pipe to process categorical features
    if cols_type:
        pipe_type = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("category", pipe_type, cols_type))

    # Establish a column transformer to process numerical and categorical features
    preprocessor = ColumnTransformer(transformers=transformers)
    # Fit and transform the data
    processed = preprocessor.fit_transform(selected_data)

    # If the processed data is a sparse matrix, convert it to a dense array
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    # Set the feature names for the processed data
    cols_names: list[str] = cols_num
    if cols_type:
        # Due to the OneHotEncoder, the feature names will be obtained throughout
        encoder = preprocessor.named_transformers_["category"]["encoder"]
        cols_names += encoder.get_feature_names_out(cols_type).tolist()

    # Convert the processed data to a DataFrame
    return DataFrame(processed, columns=cols_names), preprocessor.named_transformers_["number"]["scaler"]


def decision_boundary_adder(fig, model, X: DataFrame, pad_ratio: float = 0.05):
    """ Add decision boundary to a scatter plot.
    :param fig: the scatter plot figure
    :param model: the trained model
    :param X: the DataFrame containing the features used for training
    :param pad_ratio: the ratio to pad the decision boundary
    :return: the scatter plot figure with the decision boundary added
    """
    x_range = X.iloc[:, 0].max() - X.iloc[:, 0].min()
    y_range = X.iloc[:, 1].max() - X.iloc[:, 1].min()

    x_min = X.iloc[:, 0].min() - pad_ratio * x_range
    x_max = X.iloc[:, 0].max() + pad_ratio * x_range
    y_min = X.iloc[:, 1].min() - pad_ratio * y_range
    y_max = X.iloc[:, 1].max() + pad_ratio * y_range

    x = linspace(x_min, x_max, 100)
    y = linspace(y_min, y_max, 100)
    xx, yy = meshgrid(x, y)

    Z = model.predict(c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig.add_trace(Contour(
        x=x,
        y=y,
        z=Z,
        showscale=False,
        opacity=0.3,
        colorscale=["rgba(0,0,255,0.3)", "rgba(255,0,0,0.3)"],
        contours=dict(showlines=False),
        name="Decision Boundary"
    ))
    return fig

def console_catcher(terminal_output):
    buffer = StringIO()
