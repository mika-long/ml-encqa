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
    Generates and returns metadata for charts related to the 'filter values' task.
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
            for graph_num in range(num_graphs):
                df = generate_constrained_data(
                    num_categories=5, num_points=12, min_distance=8, rng=rng
                )
                for orientation in orientations:
                    synthetic_data_dict = create_synthetic_data_chart(
                        df=df,
                        encoding=encoding,
                        variable_type=variable_type,
                        task="filter_values",
                        graph_num=graph_num,
                        image_dir=image_dir,
                        orientation=orientation,
                        mark_type=mark_type,
                        rng=rng,
                    )
                    chart_properties = synthetic_data_dict["chart_properties"]
                    chart_out_path = synthetic_data_dict["chart_out_path"]

                    for comparison_type in task_details["comparison_type"]:
                        question, comparison_value = return_question(
                            df=df,
                            encoding=encoding,
                            variable_type=variable_type,
                            comparison_type=comparison_type,
                            mark_type=mark_type,
                            rng=rng,
                        )
                        true_label, options, comparison_type = (
                            return_answer_and_options(
                                df=df,
                                encoding=encoding,
                                variable_type=variable_type,
                                comparison_type=comparison_type,
                                comparison_value=comparison_value,
                                rng=rng,
                            )
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
                                "task": "filter_values",
                                "task_details": {
                                    "comparison_type": comparison_type,
                                    "comparison_value": str(comparison_value),
                                },
                                "encoding": encoding,
                                "variable_type": variable_type,
                                ###opinionated for now
                                "answer_type": "set",
                                "data_path": None,
                                "num_marks": chart_properties["num_marks"],
                                "num_categories": chart_properties["num_categories"],
                                "chart_spec": chart_spec,
                            }
                        )
    return chart_metadata


def return_question(
    df: pd.DataFrame,
    variable_type: str,
    encoding: str,
    comparison_type: str,
    mark_type: str,
    rng: np.random.Generator,
):
    if variable_type == "quantitative":
        sorted_cat_means = df.groupby("cat", observed=True)["var1"].mean().sort_values()
        comparison_value = (
            (sorted_cat_means.iloc[-1] + sorted_cat_means.iloc[-2]) / 2
            if rng.integers(2) == 0
            else (sorted_cat_means.iloc[0] + sorted_cat_means.iloc[1]) / 2
        )
        if comparison_type == "greater":
            if encoding == "length":
                question = f"Which bar(s) have a value of Var greater than {int(np.round(comparison_value))}?"
            elif encoding == "area":
                question = f"Which {mark_type}(s) have a value of Var greater than {int(np.round(comparison_value))}?"
            else:
                question = f"Which circle(s) have a value of Var greater than {int(np.round(comparison_value))}?"
        if comparison_type == "lesser":
            if encoding == "length":
                question = f"Which bar(s) have a value of Var less than {int(np.round(comparison_value))}?"
            elif encoding == "area":
                question = f"Which {mark_type}(s) have a value of Var less than {int(np.round(comparison_value))}?"
            else:
                question = f"Which circle(s) have a value of Var less than {int(np.round(comparison_value))}?"
    if variable_type == "nominal":
        ## get either argmax-1 or argmin+1
        num_categories = df["cat"].nunique()
        comparison_value = (
            df["cat"].value_counts().iloc[1]
            if rng.integers(2) == 0
            else df["cat"].value_counts().iloc[num_categories - 2]
        )
        if comparison_type == "greater":
            question = (
                f"Which shape(s) have more than {comparison_value} observations?"
                if encoding == "shape"
                else f"Which color(s) have more than {comparison_value} observations?"
            )
        if comparison_type == "lesser":
            question = (
                f"Which shape(s) has fewer than {comparison_value} observations?"
                if encoding == "shape"
                else f"Which color(s) have fewer than {comparison_value} observations?"
            )

    return question, comparison_value


def return_answer_and_options(
    df: pd.DataFrame,
    encoding: str,
    variable_type: str,
    comparison_type: str,
    comparison_value: float,
    rng: np.random.Generator,
):
    if variable_type == "quantitative":
        means = df.groupby("cat", observed=True)["var1"].mean()
        if comparison_type == "greater":
            answer = [k for k, v in means.items() if v > comparison_value]
        if comparison_type == "lesser":
            answer = [k for k, v in means.items() if v < comparison_value]
        answer_options = [k for k, _ in means.items()]
    if variable_type == "nominal":
        counts = df["cat"].value_counts().to_dict()
        # median_count = np.median(list(counts.values()))
        if comparison_type == "greater":
            answer = [k for k, v in counts.items() if v > comparison_value]
        if comparison_type == "lesser":
            answer = [k for k, v in counts.items() if v < comparison_value]
        answer = [
            utils.return_encoding_answer_dict(encoding=encoding)[x] for x in answer
        ]
        answer_options = [
            utils.return_encoding_answer_dict(encoding=encoding)[x]
            for x in counts.keys()
        ]
    answer_options = rng.permutation(answer_options).tolist()
    return answer, answer_options, comparison_type
