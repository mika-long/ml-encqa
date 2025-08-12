# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import pandas as pd
import numpy as np
import json
import os
from synthetic.generate_synthetic_chart import (
    create_synthetic_data_chart,
)
from synthetic.generate_synthetic_dataset import (
    generate_paired_correlated_data,
)
from scipy.stats import pearsonr


def return_chart_metadata(
    encodings: str,
    task_details: dict,
    data_dir: str,
    image_dir: str,
    rng: np.random.Generator,
    num_graphs: int,
):
    """
    Generates and returns metadata for charts related to the 'correlate values' task.
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
    ### only quantitative variables for this task
    variable_type = "quantitative"
    for encoding in encodings:
        if encoding == "area":
            mark_types = ["circle", "square"]
        else:
            mark_types = [None]
        if encoding in ["length"]:
            orientations = ["horizontal", "vertical"]
        else:
            orientations = [None]

        for mark_type in mark_types:
            for graph_num in range(num_graphs):
                df = generate_paired_correlated_data(num_points=10, rng=rng)
                for orientation in orientations:
                    synthetic_data_dict = create_synthetic_data_chart(
                        df=df,
                        encoding=encoding,
                        variable_type=variable_type,
                        task="correlate_values_relative",
                        graph_num=graph_num,
                        image_dir=image_dir,
                        orientation=orientation,
                        # fname_suffix=correlation_strength,
                        mark_type=mark_type,
                        rng=rng,
                    )
                    chart_properties = synthetic_data_dict["chart_properties"]
                    chart_out_path = synthetic_data_dict["chart_out_path"]

                    question = return_question(encoding=encoding, mark_type=mark_type)
                    true_label, options = return_answer_and_options(df=df, rng=rng)
                    chart_spec = json.dumps(chart_properties["chart_object"].to_dict())
                    chart_metadata.append(
                        {
                            "image_path": os.path.relpath(chart_out_path, data_dir),
                            "question": question,
                            "true_label": str(true_label),
                            "options": options,
                            "task": "correlate_values_relative",
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


def return_question(encoding: str, mark_type: str):
    question = ""
    if encoding == "position":
        question = "In which chart are var1 and var2 more correlated?"
    elif encoding == "area":
        question = f"In which chart are the areas of the {mark_type}s for var1 and var2 more correlated?"
    elif encoding == "length":
        question = "In which chart are the lengths of the bars for var1 and var2 more correlated?"
    elif encoding == "color_quantitative":
        question = "In which chart are the lightness of the cirlces for var1 and var2 more correlated?"

    return question


def return_answer_and_options(df: pd.DataFrame, rng: np.random.Generator):
    ### return the group with the higher correlation coefficient
    answer = (
        df.groupby("var3")
        .apply(lambda x: pearsonr(x["var1"], x["var2"])[0], include_groups=True)
        .idxmax()
    )
    answer_options = ["LEFT", "RIGHT"]
    answer_options = rng.permutation(answer_options).tolist()
    return answer, answer_options
