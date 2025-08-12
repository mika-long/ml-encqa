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
    Generates and returns metadata for charts related to the 'compute derived value exact' task.
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
            num_points_list = [6, 36] if variable_type == "nominal" else [12]
            for num_points in num_points_list:
                for graph_num in range(num_graphs):
                    df = generate_constrained_data(
                        num_categories=5,
                        num_points=num_points,
                        min_distance=5,
                        rng=rng,
                        ignore_max_min_condition=True,
                    )
                    for orientation in orientations:
                        synthetic_data_dict = create_synthetic_data_chart(
                            df=df,
                            encoding=encoding,
                            variable_type=variable_type,
                            task="compute_derived_value_exact",
                            graph_num=graph_num,
                            image_dir=image_dir,
                            orientation=orientation,
                            mark_type=mark_type,
                            rng=rng,
                            fname_suffix=(
                                f"num_points_{num_points}"
                                if variable_type == "nominal"
                                else None
                            ),
                        )
                        chart_properties = synthetic_data_dict["chart_properties"]
                        chart_out_path = synthetic_data_dict["chart_out_path"]

                        for subtask in task_details["subtask"]:
                            question = return_question(
                                encoding=encoding,
                                variable_type=variable_type,
                                subtask=subtask,
                            )
                            true_label, options = return_answer_and_options(
                                df=df,
                                variable_type=variable_type,
                                subtask=subtask,
                            )
                            chart_spec = json.dumps(
                                chart_properties["chart_object"].to_dict()
                            )
                            chart_metadata.append(
                                {
                                    "image_path": os.path.relpath(
                                        chart_out_path, data_dir
                                    ),
                                    "question": question,
                                    "true_label": str(true_label),
                                    "options": options,
                                    "task": "compute_derived_value_exact",
                                    "task_details": {"subtask": subtask},
                                    "encoding": encoding,
                                    "variable_type": variable_type,
                                    ###opinionated for now
                                    "answer_type": "numeric",
                                    "data_path": None,
                                    "num_marks": chart_properties["num_marks"],
                                    "num_categories": chart_properties[
                                        "num_categories"
                                    ],
                                    "chart_spec": chart_spec,
                                }
                            )
    return chart_metadata


def return_question(variable_type: str, encoding: str, subtask: str):
    if variable_type == "quantitative":
        if subtask == "identify":
            question = "What is the average value of Var?"
    if variable_type == "nominal":
        if subtask == "identify":
            question = (
                "What is the average number of observations per shape?"
                if encoding == "shape"
                else "What is the average number of observations per color?"
            )
    return question


def return_answer_and_options(df: pd.DataFrame, variable_type: str, subtask: str):
    if variable_type == "quantitative":
        if subtask == "identify":
            answer = df["var1"].mean()
            answer = round(answer, 2)
    if variable_type == "nominal":
        if subtask == "identify":
            answer = df["cat"].value_counts().mean()
    answer_options = None
    return answer, answer_options
