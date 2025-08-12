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
    Generates and returns metadata for charts related to the 'find extrema' task.
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
                        task="find_extrema",
                        graph_num=graph_num,
                        image_dir=image_dir,
                        orientation=orientation,
                        mark_type=mark_type,
                        rng=rng,
                    )
                    chart_properties = synthetic_data_dict["chart_properties"]
                    chart_out_path = synthetic_data_dict["chart_out_path"]

                    for extremum in task_details["extremum"]:
                        question = return_question(
                            encoding=encoding,
                            variable_type=variable_type,
                            extremum=extremum,
                            orientation=orientation,
                            mark_type=mark_type,
                        )
                        true_label, options, extremum = return_answer_and_options(
                            df=df,
                            encoding=encoding,
                            variable_type=variable_type,
                            extremum=extremum,
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
                                "task": "find_extrema",
                                "task_details": {"extremum": extremum},
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
    encoding: str, variable_type: str, extremum: str, orientation: str, mark_type: str
):
    question = ""
    if variable_type == "quantitative":
        if encoding == "position":
            if orientation == "horizontal":
                if extremum == "maximum":
                    question = "which circle is closest to the top?"
                elif extremum == "minimum":
                    question = "which circle is closest to the bottom?"
            elif orientation == "vertical":
                if extremum == "maximum":
                    question = "which circle is closest to the right?"
                elif extremum == "minimum":
                    question = "which circle is closest to the left?"
        elif encoding == "length":
            if extremum == "maximum":
                question = "which bar is the longest?"
            elif extremum == "minimum":
                question = "which bar is the shortest?"
        elif encoding == "color_quantitative":
            if extremum == "maximum":
                question = "which circle has the darkest color?"
            elif extremum == "minimum":
                question = "which circle has the lightest color?"
        elif encoding == "area":
            if extremum == "maximum":
                question = f"which {mark_type} has the largest area?"
            elif extremum == "minimum":
                question = f"which {mark_type} has the smallest area?"
    elif variable_type == "nominal":
        if encoding == "shape":
            if extremum == "maximum":
                question = "which shape has the most observations?"
            elif extremum == "minimum":
                question = "which shape has the least observations?"
        elif encoding == "color_nominal":
            if extremum == "maximum":
                question = "which color has the most observations?"
            elif extremum == "minimum":
                question = "which color has the least observations?"

    if not question:
        question = "Invalid combination of parameters"

    return question


def return_answer_and_options(
    df: pd.DataFrame,
    encoding: str,
    variable_type: str,
    extremum: str,
    rng: np.random.Generator,
):
    if (
        variable_type == "quantitative"
    ):  ### currently assumes only mean var1 is used as the encoded variable
        grouped = df.groupby("cat", observed=True)["var1"].mean().reset_index()
        max_value = grouped["var1"].max()
        min_value = grouped["var1"].min()
        if extremum == "maximum":
            answer = grouped[grouped["var1"] == max_value]["cat"].values[0]
        if extremum == "minimum":
            answer = grouped[grouped["var1"] == min_value]["cat"].values[0]
        incorrect_categories = df["cat"][df["cat"] != answer].unique()
        alternative_answers = rng.choice(incorrect_categories, size=3, replace=False)
        answer_options = [answer] + list(alternative_answers)
    elif variable_type == "nominal":
        category_counts = df.cat.value_counts().reset_index()
        if extremum == "maximum":
            answer_alphabet = category_counts[
                category_counts["count"] == category_counts["count"].max()
            ]["cat"].values[0]
        if extremum == "minimum":
            answer_alphabet = category_counts[
                category_counts["count"] == category_counts["count"].min()
            ]["cat"].values[0]
        incorrect_categories = df["cat"][df["cat"] != answer_alphabet].unique()
        alternative_answers = rng.choice(incorrect_categories, size=3, replace=False)
        answer_options_alphabets = [answer_alphabet] + list(alternative_answers)
        if encoding == "shape" or encoding == "color_nominal":
            answer = utils.return_encoding_answer_dict(encoding=encoding)[
                answer_alphabet
            ]
            answer_options = [
                utils.return_encoding_answer_dict(encoding=encoding)[x]
                for x in answer_options_alphabets
            ]
    answer_options = rng.permutation(answer_options).tolist()
    return answer, answer_options, extremum
