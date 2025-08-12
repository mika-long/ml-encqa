# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import numpy as np
import json
import os
from synthetic import utils
from synthetic.generate_synthetic_chart import (
    create_synthetic_data_chart,
)
from synthetic.generate_synthetic_dataset import (
    generate_constrained_outlier_data,
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
    Generates and returns metadata for charts related to the 'find outlier' task.
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

        if variable_type == "quantitative":
            num_categories = 5
        else:
            num_categories = 3
        for mark_type in mark_types:
            for outlier_type in task_details["outlier_type"]:
                for graph_num in range(num_graphs):
                    df = generate_constrained_outlier_data(
                        num_categories=num_categories,
                        num_points=13,
                        min_distance=8,
                        rng=rng,
                        variable_type=variable_type,
                        outlier_type=outlier_type,
                    )

                    for orientation in orientations:
                        synthetic_data_dict = create_synthetic_data_chart(
                            df=df,
                            encoding=encoding,
                            variable_type=variable_type,
                            task="find_anomaly",
                            graph_num=graph_num,
                            image_dir=image_dir,
                            mark_type=mark_type,
                            orientation=orientation,
                            rng=rng,
                            fname_suffix=f"{outlier_type}",
                        )
                        chart_properties = synthetic_data_dict["chart_properties"]
                        chart_out_path = synthetic_data_dict["chart_out_path"]

                        question = return_question(
                            encoding=encoding,
                            variable_type=variable_type,
                            mark_type=mark_type,
                        )
                        true_label, options = return_answer_and_options(
                            df=df,
                            encoding=encoding,
                            variable_type=variable_type,
                            outlier_type=outlier_type,
                            rng=rng,
                        )
                        chart_spec = json.dumps(
                            chart_properties["chart_object"].to_dict()
                        )
                        chart_metadata.append(
                            {
                                "image_path": os.path.relpath(chart_out_path, data_dir),
                                "question": question,
                                "true_label": str(true_label),
                                "options": options,
                                "task": "find_anomaly",
                                "task_details": {"outlier_type": outlier_type},
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


def return_question(encoding: str, variable_type: str, mark_type: str):
    if variable_type == "quantitative":
        if encoding == "position":
            question = (
                "which circle is an outlier relative to the rest in terms of position?"
            )
        if encoding == "length":
            question = (
                "which bar is an outlier relative to the rest in terms of length?"
            )
        if encoding == "color_quantitative":
            question = "which circle is an outlier relative to the rest in terms of color lightness?"
        if encoding == "area":
            question = f"which {mark_type} is an outlier relative to the rest in terms of area?"
    if variable_type == "nominal":
        if encoding == "shape":
            question = "which shape is an outlier relative to the rest in terms of the number of observations?"
        if encoding == "color_nominal":
            question = "which color is an outlier relative to the rest in terms of the number of observations?"
    return question


def return_answer_and_options(df, encoding, variable_type, outlier_type, rng):
    outlier_idx = None
    if variable_type == "quantitative":
        if outlier_type == "large":
            outlier_idx = df["var1"].idxmax()
        elif outlier_type == "small":
            outlier_idx = df["var1"].idxmin()
    elif variable_type == "nominal":
        if outlier_type == "large":
            outlier_idx = df["cat"].value_counts().idxmax()
        elif outlier_type == "small":
            outlier_idx = df["cat"].value_counts().idxmin()
    # elif variable_type == "nominal":
    #     outlier_idx = df["cat"].value_counts().idxmin()

    if outlier_idx is None:
        raise ValueError("Invalid combination of variable_type and outlier_type")

    true_label = (
        df.loc[outlier_idx, "cat"] if variable_type == "quantitative" else outlier_idx
    )

    options = df["cat"].unique().tolist()
    options.remove(true_label)
    rng.shuffle(options)
    options = options[:3]
    options.append(true_label)
    rng.shuffle(options)
    if variable_type == "nominal":
        true_label = utils.return_encoding_answer_dict(encoding=encoding)[true_label]
        options = [
            utils.return_encoding_answer_dict(encoding=encoding)[x] for x in options
        ]
    options = rng.permutation(options).tolist()
    return true_label, options
