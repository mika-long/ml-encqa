# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import pandas as pd
import numpy as np
import json
import os
from synthetic.generate_synthetic_chart import utils
from synthetic.generate_synthetic_chart import (
    create_synthetic_data_chart,
)
from synthetic.generate_synthetic_dataset import (
    generate_constrained_data,
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
    Generates and returns metadata for charts related to the 'retrieve value' task.
    Also saves out the chart images to the image_dir.

    Parameters:
    - encodings (str): A string of encoding types to be used for generating charts.
    - task_details (dict): A dictionary containing details about the task, such as the
      type of extremum (maximum or minimum) to find.
    - data_dir (str): The directory path where the data files are stored.
    - image_dir (str): The directory path where the generated chart images will be saved.
    - rng (np.random.Generator): A NumPy random number generator instance for generating
      random data points.
    - num_graphs (int): The number of graphs to generate for each combination of encoding
      and variable type.

    Returns:
    - list: A list of dictionaries, where each dictionary contains metadata for a single
      question
    """
    chart_metadata = []
    for encoding in encodings:
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
            for graph_num in range(num_graphs):
                df = generate_constrained_data(
                    num_categories=5, num_points=12, min_distance=8, rng=rng
                )
                for orientation in orientations:
                    synthetic_data_dict = create_synthetic_data_chart(
                        df=df,
                        encoding=encoding,
                        variable_type=variable_type,
                        task="retrieve_value",
                        graph_num=graph_num,
                        image_dir=image_dir,
                        orientation=orientation,
                        mark_type=mark_type,
                        rng=rng,
                    )

                    chart_properties = synthetic_data_dict["chart_properties"]
                    chart_out_path = synthetic_data_dict["chart_out_path"]
                    target_category = rng.choice(df["cat"].unique())
                    question = return_question(
                        encoding=encoding,
                        variable_type=variable_type,
                        target_category=target_category,
                    )
                    true_label, options, target_category = return_answer_and_options(
                        df=df,
                        encoding=encoding,
                        variable_type=variable_type,
                        target_category=target_category,
                    )
                    chart_spec = json.dumps(chart_properties["chart_object"].to_dict())
                    chart_metadata.append(
                        {
                            "image_path": os.path.relpath(chart_out_path, data_dir),
                            "question": question,
                            "true_label": str(true_label),
                            "options": options,
                            "task": "retrieve_value",
                            "task_details": {"target_category": target_category},
                            "encoding": encoding,
                            "variable_type": variable_type,
                            ###opinionated for now
                            "answer_type": "numeric",
                            "data_path": None,
                            "num_marks": chart_properties["num_marks"],
                            "num_categories": chart_properties["num_categories"],
                            "chart_spec": chart_spec,
                        }
                    )
    return chart_metadata


def return_question(variable_type: str, encoding: str, target_category: str):
    if variable_type == "quantitative":
        question = f"What is the value of Var at {target_category}?"
    if variable_type == "nominal":
        question = (
            f"How many {utils.return_encoding_answer_dict(encoding=encoding)[target_category]} shapes are present?"
            if encoding == "shape"
            else f"How many {utils.return_encoding_answer_dict(encoding=encoding)[target_category]} circles are present?"
        )
    return question


def return_answer_and_options(
    df: pd.DataFrame, encoding: str, variable_type: str, target_category: str
):
    if variable_type == "quantitative":
        answer = df[df["cat"] == target_category]["var1"].mean()
        answer = round(answer, 2)
        answer_options = None
    if variable_type == "nominal":
        answer = df[df["cat"] == target_category].shape[0]
        if encoding == "shape" or encoding == "color_nominal":
            target_category = utils.return_encoding_answer_dict(encoding=encoding)[
                target_category
            ]
        answer_options = None
    return answer, answer_options, target_category
