# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import pandas as pd
import altair as alt
from altair.utils.data import to_values
import os

# from synthetic.generate_synthetic_dataset import generate_constrained_data
import numpy as np
from collections import Counter, defaultdict
from synthetic import utils

# each task has its own counter for color scheme
color_scheme_counters: defaultdict = defaultdict(Counter)
chart_width = 400
chart_height = 400


def get_legend_values(df):
    values = [
        df.groupby("cat", observed=True)["var1"].mean().quantile(0.10),
        df.groupby("cat", observed=True)["var1"].mean().quantile(0.50),
        df.groupby("cat", observed=True)["var1"].mean().quantile(1) * 1.1,
    ]
    return values


def get_legend_direction(rng: np.random.Generator):
    return "horizontal" if rng.integers(0, 2) == 0 else "vertical"


def get_assigned_color_scheme(task: str):
    """
    function to help check the position of a number in a list of 4
    in order to assign a color scheme to the chart.
    Every 4 charts we cycle through the color schemes."""
    counter: Counter = color_scheme_counters[task]
    position = counter["assigned_color_scheme"] % 4
    counter["assigned_color_scheme"] += 1

    # position = (number+1) % 4
    if position == 0:
        return "greys"
    elif position == 1:
        return "blues"
    elif position == 2:
        return "greens"
    elif position == 3:
        return "reds"


def length_chart(df: pd.DataFrame, task: str, orientation: str):
    encoding = {}
    if task != "correlate_values":
        data_dict = to_values(df)
        axis_variables = {
            "var1": {
                "field": "var1",
                "type": "quantitative",
                "aggregate": "mean",
                "axis": (
                    {"labels": False, "title": None}
                    if task in ["find_extrema", "find_anomaly"]
                    else {"title": "Var"}
                ),
            },
            "cat": {"field": "cat", "type": "nominal", "axis": {"title": None}},
        }
        if orientation == "vertical":
            encoding["x"], encoding["y"] = axis_variables["var1"], axis_variables["cat"]

        if orientation == "horizontal":
            encoding["y"], encoding["x"] = axis_variables["var1"], axis_variables["cat"]
            encoding["x"]["axis"]["labelAngle"] = 0

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "bar"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
        }
    elif task == "correlate_values":
        n = df.shape[0]
        # Preparing the data in long format for Altair
        index = np.arange(n)
        long_df = pd.DataFrame(
            {
                "category": np.tile(index, 2),
                "group": ["var1"] * n + ["var2"] * n,
                "values": np.concatenate([df["var1"].values, df["var2"].values]),
            }
        )
        data_dict = to_values(long_df)
        axis_variables = {
            "category": {
                "field": "category",
                "type": "nominal",
                "axis": {"title": None},
            },
            "values": {
                "field": "values",
                "type": "quantitative",
                "axis": {"title": None, "labels": False},
            },
            "xoffset": {"field": "group", "type": "nominal"},
            "yoffset": {"field": "group", "type": "nominal"},
        }
        if orientation == "vertical":
            encoding["x"], encoding["y"], encoding["xOffset"] = (
                axis_variables["category"],
                axis_variables["values"],
                axis_variables["xoffset"],
            )
            encoding["x"]["axis"]["labelAngle"] = 0

        if orientation == "horizontal":
            encoding["y"], encoding["x"], encoding["yOffset"] = (
                axis_variables["category"],
                axis_variables["values"],
                axis_variables["yoffset"],
            )

        encoding["color"] = {"field": "group", "type": "nominal"}

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "bar"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
        }
    return chart_spec


def position_encoding_chart(df: pd.DataFrame, task: str, orientation: str):
    """
    orientation: horizontal or vertical
    """
    encoding = {}
    if task != "correlate_values":
        data_dict = to_values(df)
        axis_variables = {
            "var1": {
                "field": "var1",
                "type": "quantitative",
                "aggregate": "mean",
                "axis": (
                    {"labels": False, "title": None}
                    if task in ["find_extrema", "find_anomaly"]
                    else {"title": "Var"}
                ),
            },
            "cat": {"field": "cat", "type": "nominal", "axis": {"title": None}},
        }

        # Swap x and y encodings if orientation is horizontal
        if orientation == "vertical":
            encoding["x"], encoding["y"] = axis_variables["var1"], axis_variables["cat"]

        if orientation == "horizontal":
            encoding["y"], encoding["x"] = axis_variables["var1"], axis_variables["cat"]
            encoding["x"]["axis"]["labelAngle"] = 0

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "point", "filled": True, "size": 300, "color": "gray"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
        }
    elif task == "correlate_values":
        data_dict = to_values(df)
        axis_variables = {
            "var1": {
                "field": "var1",
                "type": "quantitative",
                "axis": {"labels": False},
                "scale": {"domain": [df.var1.min() - 1, df.var1.max() + 1]},
            },
            "var2": {
                "field": "var2",
                "type": "quantitative",
                "axis": {"labels": False},
                "scale": {"domain": [df.var2.min() - 1, df.var2.max() + 1]},
            },
        }

        encoding["x"], encoding["y"] = axis_variables["var1"], axis_variables["var2"]

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "point", "filled": True, "size": 300, "color": "gray"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
        }

    return chart_spec


def area_encoding_chart(
    df: pd.DataFrame, task: str, mark_type: str, rng: np.random.Generator
):
    legend_direction = get_legend_direction(rng)
    if task != "correlate_values":
        data_dict = to_values(df)
        custom_scale = alt.Scale(
            domain=[
                df.groupby("cat", observed=True)["var1"].mean().min(),
                df.groupby("cat", observed=True)["var1"].mean().max(),
            ],
            range=[500, 5000],
        )
        chart_spec = {
            "data": data_dict,
            "mark": {"type": mark_type, "filled": True, "size": 300, "color": "gray"},
            "encoding": {
                "size": {
                    "field": "var1",
                    "aggregate": "mean",
                    "type": "quantitative",
                    "scale": custom_scale.to_dict() if custom_scale else {},
                    "legend": (
                        None
                        if task in ["find_extrema", "find_anomaly"]
                        else {
                            "title": "Var",
                            "orient": "right",
                            "direction": legend_direction,
                            "values": get_legend_values(df),
                        }
                    ),
                },
                "y": {"field": "cat", "type": "nominal", "axis": {"title": None}},
            },
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}},
        }
    elif task == "correlate_values":
        n = df.shape[0]
        # Preparing the data in long format for Altair
        index = np.arange(n)
        long_df = pd.DataFrame(
            {
                "category": np.tile(index, 2),
                "group": ["var1"] * n + ["var2"] * n,
                "values": np.concatenate([df["var1"].values, df["var2"].values]),
            }
        )
        custom_scale = alt.Scale(
            domain=[
                long_df["values"].min(),
                long_df["values"].max(),
            ],
            range=[50, 500],
        )
        data_dict = to_values(long_df)

        chart_spec = {
            "data": data_dict,
            "mark": {"type": mark_type, "filled": True, "size": 300, "color": "gray"},
            "encoding": {
                "size": {
                    "field": "values",
                    "type": "quantitative",
                    "scale": custom_scale.to_dict() if custom_scale else {},
                    "legend": (None),
                },
                "y": {
                    "field": "category",
                    "type": "nominal",
                    "axis": {"title": None},
                },
                "x": {
                    "field": "group",
                    "type": "nominal",
                    "axis": {"title": None, "labelAngle": 0},
                },
            },
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}},
        }
    return chart_spec


def color_encoding_chart(
    df: pd.DataFrame, task: str, variable_type: str, rng: np.random.Generator
):
    color_scheme = get_assigned_color_scheme(task)
    legend_direction = get_legend_direction(rng)
    data_dict = to_values(df)
    if variable_type == "quantitative":
        if task != "correlate_values":
            chart_spec = {
                "data": data_dict,
                "mark": {"type": "point", "filled": True, "size": 300},
                "encoding": {
                    "color": {
                        "field": "var1",
                        "aggregate": "mean",
                        "type": "quantitative",
                        "legend": (
                            None
                            if task in ["find_extrema", "find_anomaly"]
                            else {
                                "title": "Var",
                                "orient": "right",
                                "direction": legend_direction,
                            }
                        ),
                        "scale": {
                            "domain": [df.var1.min(), df.var1.max()],
                            "scheme": color_scheme,
                        },
                    },
                    "y": {"field": "cat", "type": "nominal", "axis": {"title": None}},
                },
                "width": chart_width,
                "height": chart_height,
                "config": {"axis": {"grid": False}},
            }
        elif task == "correlate_values":
            n = df.shape[0]
            # Preparing the data in long format for Altair
            index = np.arange(n)
            long_df = pd.DataFrame(
                {
                    "category": np.tile(index, 2),
                    "group": ["var1"] * n + ["var2"] * n,
                    "values": np.concatenate(
                        [df["var1"].values, df["var2"].values]
                    ),  # Values for x and y
                }
            )
            data_dict = to_values(long_df)
            custom_scale = alt.Scale(
                domain=[
                    long_df["values"].min(),
                    long_df["values"].max(),
                ],
                scheme=color_scheme,
            )
            chart_spec = {
                "data": data_dict,
                "mark": {"type": "point", "filled": True, "size": 300},
                "encoding": {
                    "color": {
                        "field": "values",
                        "type": "quantitative",
                        "scale": custom_scale.to_dict() if custom_scale else {},
                        "legend": None,
                    },
                    "y": {
                        "field": "category",
                        "type": "nominal",
                        "axis": {"title": None},
                    },
                    # "xOffset": {"field": "group", "type": "nominal"},
                    "x": {
                        "field": "group",
                        "type": "nominal",
                        "axis": {"title": None, "labelAngle": 0},
                    },
                },
                "width": chart_width,
                "height": chart_height,
                "config": {"axis": {"grid": False}},
            }

    if variable_type == "nominal":
        domain = ["A", "B", "C", "D", "E", "F"]
        custom_scale = alt.Scale(
            domain=domain,
            range=[
                utils.return_encoding_answer_dict("color_nominal", vega_lite_spec=True)[
                    x
                ]
                for x in domain
            ],
        )

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "point", "filled": True, "size": 300},
            "encoding": {
                "x": {
                    "field": "var1",
                    "type": "quantitative",
                    "scale": {"domain": [df.var1.min() - 10, df.var1.max() + 10]},
                    "axis": {"labels": False, "title": None},
                },
                "y": {
                    "field": "var2",
                    "type": "quantitative",
                    "scale": {"domain": [df.var2.min() - 10, df.var2.max() + 10]},
                    "axis": {"labels": False, "title": None},
                },
                "color": {
                    "field": "cat",
                    "type": "nominal",
                    "legend": None,
                    "scale": custom_scale.to_dict() if custom_scale else {},
                },
            },
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}},
        }
    return chart_spec


def shape_encoding_chart(df: pd.DataFrame):
    data_dict = to_values(df)
    domain = ["A", "B", "C", "D", "E", "F"]
    custom_scale = alt.Scale(
        domain=domain,
        range=[
            utils.return_encoding_answer_dict("shape", vega_lite_spec=True)[x]
            for x in domain
        ],
    )
    chart_spec = {
        "data": data_dict,
        "mark": {"type": "point", "filled": True, "size": 300, "color": "gray"},
        "encoding": {
            "x": {
                "field": "var1",
                "type": "quantitative",
                "scale": {"domain": [df.var1.min() - 10, df.var1.max() + 10]},
                "axis": {"labels": False, "title": None},
            },
            "y": {
                "field": "var2",
                "type": "quantitative",
                "scale": {"domain": [df.var2.min() - 10, df.var2.max() + 10]},
                "axis": {"labels": False, "title": None},
            },
            "shape": {
                "field": "cat",
                "type": "nominal",
                "scale": custom_scale.to_dict() if custom_scale else {},
                "legend": None,
            },
        },
        "width": chart_width,
        "height": chart_height,
        "config": {"axis": {"grid": False}},
    }
    return chart_spec


def paired_length_encoding_chart(df: pd.DataFrame, task: str, orientation: str):
    encoding = {}
    if task != "correlate_values_relative":
        n = df.shape[0]
        index = np.arange(n)
        long_df = pd.DataFrame(
            {
                "category": np.tile(
                    index, 2
                ),  # Repeat each category twice, once for x and once for y
                "group": ["LEFT"] * n + ["RIGHT"] * n,
                "values": np.concatenate([df["var1"].values, df["var2"].values]),
            }
        )
        data_dict = to_values(long_df)
        axis_variables = {
            "category": {
                "field": "category",
                "type": "nominal",
                "axis": {"title": None},
            },
            "values": {
                "field": "values",
                "type": "quantitative",
                "axis": {"title": "Var", "labels": False},
            },
        }
        if orientation == "vertical":
            encoding["x"], encoding["y"] = (
                axis_variables["category"],
                axis_variables["values"],
            )
            encoding["x"]["axis"]["labelAngle"] = 0

        if orientation == "horizontal":
            encoding["y"], encoding["x"] = (
                axis_variables["category"],
                axis_variables["values"],
            )
            encoding["x"]["axis"]["labelAngle"] = 0
        encoding["column"] = {"field": "group", "type": "nominal", "title": None}

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "bar"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
        }
    elif task == "correlate_values_relative":
        n = df.shape[0] // 2
        index = np.arange(n)
        long_df = pd.melt(df, id_vars="var3", var_name="group", value_name="values")
        long_df["category"] = np.tile(index, 4)
        data_dict = to_values(long_df)
        axis_variables = {
            "category": {
                "field": "category",
                "type": "nominal",
                "axis": {"title": None},
            },
            "values": {
                "field": "values",
                "type": "quantitative",
                "axis": {"title": None, "labels": False},
            },
            "xoffset": {"field": "group", "type": "nominal"},
            "yoffset": {"field": "group", "type": "nominal"},
        }
        if orientation == "vertical":
            encoding["x"], encoding["y"], encoding["xOffset"] = (
                axis_variables["category"],
                axis_variables["values"],
                axis_variables["xoffset"],
            )
            encoding["x"]["axis"]["labelAngle"] = 0

        if orientation == "horizontal":
            encoding["y"], encoding["x"], encoding["yOffset"] = (
                axis_variables["category"],
                axis_variables["values"],
                axis_variables["yoffset"],
            )

        encoding["color"] = {"field": "group", "type": "nominal"}
        encoding["column"] = {"field": "var3", "type": "nominal", "title": None}

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "bar"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
        }

    return chart_spec


def paired_position_encoding_chart(df: pd.DataFrame, task: str, orientation: str):
    encoding = {}
    if task != "correlate_values_relative":
        n = df.shape[0]
        index = np.arange(n)
        long_df = pd.DataFrame(
            {
                "category": np.tile(
                    index, 2
                ),  # Repeat each category twice, once for x and once for y
                "group": ["LEFT"] * n + ["RIGHT"] * n,
                "values": np.concatenate([df["var1"].values, df["var2"].values]),
            }
        )
        data_dict = to_values(long_df)
        axis_variables = {
            "category": {
                "field": "category",
                "type": "nominal",
                "axis": {"title": None},
            },
            "values": {
                "field": "values",
                "type": "quantitative",
                "axis": {"title": "Var", "labels": False},
            },
        }

        if orientation == "vertical":
            encoding["x"], encoding["y"] = (
                axis_variables["category"],
                axis_variables["values"],
            )
            encoding["x"]["axis"]["labelAngle"] = 0

        if orientation == "horizontal":
            encoding["y"], encoding["x"] = (
                axis_variables["category"],
                axis_variables["values"],
            )
            encoding["x"]["axis"]["labelAngle"] = 0
        encoding["column"] = {"field": "group", "type": "nominal", "title": None}

        chart_spec = {
            "data": data_dict,
            "mark": {"type": "point", "filled": True, "size": 300, "color": "gray"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
        }
    elif task == "correlate_values_relative":
        data_dict = to_values(df)
        axis_variables = {
            "var1": {
                "field": "var1",
                "type": "quantitative",
                "axis": {"labels": False},
                "scale": {"domain": [df.var1.min() - 1, df.var1.max() + 1]},
            },
            "var2": {
                "field": "var2",
                "type": "quantitative",
                "axis": {"labels": False},
                "scale": {"domain": [df.var2.min() - 1, df.var2.max() + 1]},
            },
        }
        encoding["x"], encoding["y"] = axis_variables["var1"], axis_variables["var2"]
        encoding["column"] = {"field": "var3", "type": "nominal", "title": None}
        chart_spec = {
            "data": data_dict,
            "mark": {"type": "point", "filled": True, "size": 300, "color": "gray"},
            "encoding": encoding,
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}, "bar": {"color": "gray"}},
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
        }

    return chart_spec


def paired_area_encoding_chart(df: pd.DataFrame, task: str, mark_type: str):
    if task != "correlate_values_relative":
        n = df.shape[0]
        index = np.arange(n)
        long_df = pd.DataFrame(
            {
                "category": np.tile(
                    index, 2
                ),  # Repeat each category twice, once for x and once for y
                "group": ["LEFT"] * n + ["RIGHT"] * n,
                "values": np.concatenate([df["var1"].values, df["var2"].values]),
            }
        )
        data_dict = to_values(long_df)
        custom_scale = alt.Scale(
            domain=[
                long_df["values"].min(),
                long_df["values"].max(),
            ],
            range=[400, 3500],
        )
        chart_spec = {
            "data": data_dict,
            "mark": {"type": mark_type, "filled": True, "size": 300, "color": "gray"},
            "encoding": {
                "size": {
                    "field": "values",
                    "type": "quantitative",
                    "scale": custom_scale.to_dict() if custom_scale else {},
                    "legend": None,
                },
                "y": {"field": "category", "type": "nominal", "axis": {"title": None}},
                "column": {"field": "group", "type": "nominal", "title": None},
            },
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}},
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
        }
    elif task == "correlate_values_relative":
        n = df.shape[0] // 2
        index = np.arange(n)
        long_df = pd.melt(df, id_vars="var3", var_name="group", value_name="values")
        long_df["category"] = np.tile(index, 4)
        data_dict = to_values(long_df)
        custom_scale = alt.Scale(
            domain=[
                long_df["values"].min(),
                long_df["values"].max(),
            ],
            range=[50, 500],
        )
        data_dict = to_values(long_df)

        chart_spec = {
            "data": data_dict,
            "mark": {"type": mark_type, "filled": True, "size": 300, "color": "gray"},
            "encoding": {
                "size": {
                    "field": "values",
                    # "aggregate": "mean",
                    "type": "quantitative",
                    "scale": custom_scale.to_dict() if custom_scale else {},
                    "legend": (None),
                },
                "y": {
                    "field": "category",
                    "type": "nominal",
                    "axis": {"title": None},
                },
                "x": {
                    "field": "group",
                    "type": "nominal",
                    "axis": {"title": None, "labelAngle": 0},
                },
                "column": {"field": "var3", "type": "nominal", "title": None},
            },
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}},
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
        }
    return chart_spec


def paired_color_encoding_chart(df: pd.DataFrame, task: str, variable_type: str):
    color_scheme = get_assigned_color_scheme(task)
    if variable_type == "quantitative":
        if task != "correlate_values_relative":
            n = df.shape[0]
            index = np.arange(n)
            long_df = pd.DataFrame(
                {
                    "category": np.tile(
                        index, 2
                    ),  # Repeat each category twice, once for x and once for y
                    "group": ["LEFT"] * n + ["RIGHT"] * n,
                    "values": np.concatenate([df["var1"].values, df["var2"].values]),
                }
            )
            data_dict = to_values(long_df)
            custom_scale = alt.Scale(
                domain=[
                    long_df["values"].min(),
                    long_df["values"].max(),
                ],
                scheme=color_scheme,
            )
            chart_spec = {
                "data": data_dict,
                "mark": {"type": "point", "filled": True, "size": 300},
                "encoding": {
                    "color": {
                        "field": "values",
                        "type": "quantitative",
                        "scale": custom_scale.to_dict() if custom_scale else {},
                        "legend": None,
                    },
                    "y": {
                        "field": "category",
                        "type": "nominal",
                        "axis": {"title": None},
                    },
                    "column": {"field": "group", "type": "nominal", "title": None},
                },
                "width": 300,
                "height": 400,
                "config": {"axis": {"grid": False}},
                "resolve": {"axis": {"x": "independent", "y": "independent"}},
            }
        elif task == "correlate_values_relative":
            n = df.shape[0] // 2
            index = np.arange(n)
            long_df = pd.melt(df, id_vars="var3", var_name="group", value_name="values")
            long_df["category"] = np.tile(index, 4)
            data_dict = to_values(long_df)
            data_dict = to_values(long_df)
            custom_scale = alt.Scale(
                domain=[
                    long_df["values"].min(),
                    long_df["values"].max(),
                ],
                scheme=color_scheme,
            )
            chart_spec = {
                "data": data_dict,
                "mark": {"type": "point", "filled": True, "size": 300},
                "encoding": {
                    "color": {
                        "field": "values",
                        "type": "quantitative",
                        "scale": custom_scale.to_dict() if custom_scale else {},
                        "legend": None,
                    },
                    "y": {
                        "field": "category",
                        "type": "nominal",
                        "axis": {"title": None},
                    },
                    "x": {
                        "field": "group",
                        "type": "nominal",
                        "axis": {"title": None, "labelAngle": 0},
                    },
                    "column": {"field": "var3", "type": "nominal", "title": None},
                },
                "width": chart_width,
                "height": chart_height,
                "config": {"axis": {"grid": False}},
                "resolve": {"axis": {"x": "independent", "y": "independent"}},
            }
    if variable_type == "nominal":
        data_dict = to_values(df)
        domain = ["A", "B", "C", "D", "E", "F"]
        custom_scale = alt.Scale(
            domain=domain,
            range=[
                utils.return_encoding_answer_dict("color_nominal", vega_lite_spec=True)[
                    x
                ]
                for x in domain
            ],
        )
        chart_spec = {
            "data": data_dict,
            "mark": {"type": "point", "filled": True, "size": 300},
            "encoding": {
                "x": {
                    "field": "var1",
                    "type": "quantitative",
                    "scale": {"domain": [df.var1.min() - 10, df.var1.max() + 10]},
                    "axis": {"labels": False, "title": None},
                },
                "y": {
                    "field": "var2",
                    "type": "quantitative",
                    "scale": {"domain": [df.var2.min() - 10, df.var2.max() + 10]},
                    "axis": {"labels": False, "title": None},
                },
                "color": {
                    "field": "cat",
                    "type": "nominal",
                    "legend": None,
                    "scale": custom_scale.to_dict() if custom_scale else {},
                },
                "column": {"field": "var3", "type": "nominal", "title": None},
            },
            "width": chart_width,
            "height": chart_height,
            "config": {"axis": {"grid": False}},
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
        }

    return chart_spec


def paired_shape_encoding_chart(df: pd.DataFrame):
    data_dict = to_values(df)
    domain = ["A", "B", "C", "D", "E", "F"]
    custom_scale = alt.Scale(
        domain=domain,
        range=[
            utils.return_encoding_answer_dict("shape", vega_lite_spec=True)[x]
            for x in domain
        ],
    )
    chart_spec = {
        "data": data_dict,
        "mark": {"type": "point", "filled": True, "size": 300, "color": "gray"},
        "encoding": {
            "x": {
                "field": "var1",
                "type": "quantitative",
                "scale": {"domain": [df.var1.min() - 10, df.var1.max() + 10]},
                "axis": {"labels": False, "title": None},
            },
            "y": {
                "field": "var2",
                "type": "quantitative",
                "scale": {"domain": [df.var2.min() - 10, df.var2.max() + 10]},
                "axis": {"labels": False, "title": None},
            },
            "shape": {
                "field": "cat",
                "type": "nominal",
                "legend": None,
                "scale": custom_scale.to_dict() if custom_scale else {},
            },
            "column": {"field": "var3", "type": "nominal", "title": None},
        },
        "width": chart_width,
        "height": chart_height,
        "config": {"axis": {"grid": False}},
        "resolve": {"axis": {"x": "independent", "y": "independent"}},
    }
    return chart_spec


def create_altair_chart(
    df: pd.DataFrame,
    encoding: str,
    variable_type: str,
    task: str,
    orientation: str,
    mark_type: str,
    rng: np.random.Generator,
):
    """
    Create an Altair chart from the given DataFrame and return the chart, number of marks, and number of categories.
    """
    ### check if cat is in the columns for df
    if "cat" not in df.columns:
        num_categories = 0
    else:
        num_categories = df.cat.nunique()
    if variable_type == "nominal":
        num_marks = df.shape[0]
    elif variable_type == "quantitative" and task not in [
        "correlate_values",
        "correlate_values_relative",
    ]:
        num_marks = df.cat.nunique()
    elif task in ["correlate_values", "correlate_values_relative"]:
        num_marks = df.shape[0] * 2

    if task not in ["compute_derived_value_relative", "correlate_values_relative"]:
        if encoding == "length":
            chart_spec = length_chart(df, task, orientation)
        if encoding == "position":
            chart_spec = position_encoding_chart(df, task, orientation)
        if encoding == "area":
            chart_spec = area_encoding_chart(df, task, mark_type, rng)
        if encoding in ["color_nominal", "color_quantitative"]:
            chart_spec = color_encoding_chart(df, task, variable_type, rng)
        if encoding == "shape":
            chart_spec = shape_encoding_chart(df)
        chart_object = alt.Chart.from_dict(chart_spec)
    elif task in ["compute_derived_value_relative", "correlate_values_relative"]:
        if encoding == "length":
            chart_spec = paired_length_encoding_chart(df, task, orientation)
        if encoding == "position":
            chart_spec = paired_position_encoding_chart(df, task, orientation)
        if encoding == "area":
            chart_spec = paired_area_encoding_chart(df, task, mark_type)
        if encoding in ["color_nominal", "color_quantitative"]:
            chart_spec = paired_color_encoding_chart(df, task, variable_type)
        if encoding == "shape":
            chart_spec = paired_shape_encoding_chart(df)
        chart_object = alt.Chart.from_dict(chart_spec)

    return {
        "chart_object": chart_object,
        "num_marks": num_marks,
        "num_categories": num_categories,
    }


def create_synthetic_data_chart(
    df: pd.DataFrame,
    encoding,
    variable_type,
    task,
    graph_num,
    image_dir,
    rng: np.random.Generator,
    orientation=None,
    **kwargs,
):
    """
    This function creates a random dataset based on the input parameters and generates an altair chart object
    The chart is saved out to a file in the image_dir/task/encoding directory and the chart properties are returned
    """
    fname_suffix = kwargs.get("fname_suffix", None)
    mark_type = kwargs.get("mark_type", None)
    ### return the chart object and the number of marks and categories
    chart_properties = create_altair_chart(
        df=df,
        encoding=encoding,
        variable_type=variable_type,
        task=task,
        orientation=orientation,
        mark_type=mark_type,
        rng=rng,
    )
    ### make a filename for the chart image
    # if encoding == "color":
    #     chart_fname = f"{graph_num}_{variable_type}.png"
    if encoding in ["position", "length"]:
        chart_fname = f"{graph_num}_{orientation}.png"
    elif encoding == "area":
        chart_fname = f"{graph_num}_{mark_type}.png"
    else:
        chart_fname = f"{graph_num}.png"
    if fname_suffix:
        chart_fname = chart_fname.replace(".png", f"_{fname_suffix}.png")
    chart_out_dir = os.path.join(image_dir, task, encoding)
    os.makedirs(chart_out_dir, exist_ok=True)
    ### save the chart image
    chart_out_path = os.path.join(chart_out_dir, chart_fname)
    chart_properties["chart_object"].save(chart_out_path)

    return {
        "chart_properties": chart_properties,
        "chart_out_path": chart_out_path,
    }
