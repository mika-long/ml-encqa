# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import pandas as pd
import numpy as np
import json
import os
from synthetic import utils
from synthetic.generate_synthetic_chart import (
    create_synthetic_data_chart,
)
from synthetic.generate_synthetic_dataset import (
    generate_paired_data,
)


def return_chart_metadata(
    encodings: str,
    task_details: dict,
    data_dir: str,
    image_dir: str,
    rng: np.random.Generator,
    num_graphs: int,
):
    """
    Generates and returns metadata for charts related to the 'compute derived value relative' task.
    Also saves out the chart images to the image_dir.

    Parameters:
    - encodings (str): A string of encoding types to be used for generating charts.
    - task_details (dict): A dictionary containing details about the task
    - data_dir (str): The directory path where the data files are stored.
    - image_dir (str): The directory path where the generated chart images will be saved.
    - rng (np.random.Generator): A NumPy random number generator instance for generating
    """
    chart_metadata = []
    for encoding in encodings:
        ### get variable types depending on the encoding
        variable_type = utils.get_encoding_variable_types(encoding)
        if encoding == "area":
            mark_types = ["circle", "square"]
        else:
            mark_types = [None]

        if encoding in ["position", "length"]:
            orientations = ["horizontal", "vertical"]
        else:
            orientations = [None]
        for mark_type in mark_types:
            num_categories = [3, 5] if variable_type == "nominal" else [5]
            num_points = 15 if variable_type == "nominal" else 6
            for graph_num in range(num_graphs):
                df = generate_paired_data(
                    num_categories=num_categories,
                    num_points=num_points,
                    min_distance=6,
                    rng=rng,
                    variable_type=variable_type,
                )
                for orientation in orientations:
                    synthetic_data_dict = create_synthetic_data_chart(
                        df=df,
                        encoding=encoding,
                        variable_type=variable_type,
                        task="compute_derived_value_relative",
                        graph_num=graph_num,
                        image_dir=image_dir,
                        orientation=orientation,
                        mark_type=mark_type,
                        rng=rng,
                    )
                    chart_properties = synthetic_data_dict["chart_properties"]
                    chart_out_path = synthetic_data_dict["chart_out_path"]

                    question = return_question(
                        encoding=encoding,
                        variable_type=variable_type,
                        orientation=orientation,
                        mark_type=mark_type,
                    )
                    true_label, options = return_answer_and_options(
                        df=df, variable_type=variable_type, rng=rng
                    )
                    chart_spec = json.dumps(chart_properties["chart_object"].to_dict())
                    chart_metadata.append(
                        {
                            "image_path": os.path.relpath(chart_out_path, data_dir),
                            "question": question,
                            "true_label": str(true_label),
                            "options": options,
                            "task": "compute_derived_value_relative",
                            "task_details": {},
                            "encoding": encoding,
                            "variable_type": variable_type,
                            ###opinionated for now
                            "answer_type": "multiple_choice",
                            "data_path": None,
                            "num_marks": chart_properties["num_marks"],
                            "num_categories": chart_properties["num_categories"],
                            "chart_spec": chart_spec,
                        }
                    )
    return chart_metadata


def return_question(
    variable_type: str, encoding: str, orientation: str, mark_type: str
):
    if variable_type == "quantitative":
        if encoding == "length":
            question = "In which chart are the bars longer on average?"
        if encoding == "position":
            if orientation == "horizontal":
                question = (
                    "In which chart are the circles further to the right on average?"
                )
            if orientation == "vertical":
                question = (
                    "In which chart are the circles further to the top on average?"
                )
        if encoding == "area":
            question = (
                f"In which chart are the areas of the {mark_type}s larger on average?"
            )
        if encoding == "color_quantitative":
            question = "In which chart are the circles darker on average?"
    if variable_type == "nominal":
        question = (
            "Which chart has a higher average number of observations per shape?"
            if encoding == "shape"
            else "Which chart has a higher average number of observations per colors"
        )
    return question


def return_answer_and_options(
    df: pd.DataFrame, variable_type: str, rng: np.random.Generator
):
    if variable_type == "quantitative":
        if df["var1"].mean() > df["var2"].mean():
            answer = "LEFT"
        else:
            answer = "RIGHT"

    if variable_type == "nominal":
        total_points_per_group = df.groupby("var3")["cat"].size()
        unique_cats_per_group = df.groupby("var3")["cat"].nunique()
        mean_points_per_category = total_points_per_group / unique_cats_per_group
        answer = mean_points_per_category.idxmax()
    answer_options = ["LEFT", "RIGHT"]
    answer_options = rng.permutation(answer_options).tolist()
    return answer, answer_options
