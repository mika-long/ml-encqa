# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import os
import numpy as np
from absl import app, flags  # type: ignore

from synthetic import (
    find_extrema,
    retrieve_value,
    filter_values,
    compute_derived_value_exact,
    compute_derived_value_relative,
    find_anomaly,
    correlate_values,
    correlate_values_relative,
)

"""
Usage: python enc_qa/synthetic/generate_synthetic.py --TASK=task_name  --NUM_GRAPHS=num_graphs
"""
FLAGS = flags.FLAGS

flags.DEFINE_list(
    "ENCODING",
    ["color_nominal", "color_quantitative", "position", "area", "shape", "length"],
    "encoding visual feature",
)

flags.DEFINE_list(
    "TASK",
    [
        "find_extrema",
        "retrieve_value",
        "filter_values",
        "compute_derived_value_exact",
        "find_anomaly",
        "correlate_values",
        "compute_derived_value_relative",
        "correlate_values_relative",
    ],
    "task to be performed on the graph",
)
flags.DEFINE_integer(
    "NUM_GRAPHS",
    25,
    "number of graphs to generate",
)
flags.DEFINE_string(
    "TARGET_DIR",
    "data/",
    "directory to save the data to",
)

flags.DEFINE_integer(
    "SEED",
    42,
    "random seed for reproducibility",
)
task_details_dict = {
    "find_extrema": {
        "extremum": ["maximum", "minimum"],
    },
    "retrieve_value": {},
    "compute_derived_value_exact": {"subtask": ["identify"]},
    "filter_values": {
        "comparison_type": ["greater", "lesser"],
    },
    "find_anomaly": {"outlier_type": ["small", "large"]},
    "correlate_values": {
        "correlation_strength": ["weak", "strong"],
    },
    "compute_derived_value_relative": {},
    "correlate_values_relative": {},
}

task_function_dict = {
    "find_extrema": find_extrema.return_chart_metadata,
    "retrieve_value": retrieve_value.return_chart_metadata,
    "compute_derived_value_exact": compute_derived_value_exact.return_chart_metadata,
    "filter_values": filter_values.return_chart_metadata,
    "find_anomaly": find_anomaly.return_chart_metadata,
    "correlate_values": correlate_values.return_chart_metadata,
    "compute_derived_value_relative": compute_derived_value_relative.return_chart_metadata,
    "correlate_values_relative": correlate_values_relative.return_chart_metadata,
}


def generate_charts_and_metadata(
    encodings, task, num_graphs, data_dir, image_dir, seed
):
    """
    Generate num_graphs charts and associated metadata for the given encoding, variable type, task, question type, and save the images to the appropriate subfolder in image_dir.
    """
    chart_metadata = []
    task_function = task_function_dict[task]
    task_details = task_details_dict[task]
    chart_metadata = task_function(
        encodings=encodings,
        task_details=task_details,
        data_dir=data_dir,
        image_dir=image_dir,
        #### ensure each task starts with the exact same seed
        rng=np.random.default_rng(seed),
        num_graphs=num_graphs,
    )

    return chart_metadata


def main(argv):
    del argv
    ### root data directory
    target_dir = FLAGS.TARGET_DIR
    image_dir = os.path.join(target_dir, "synthetic_data", "images")
    os.makedirs(image_dir, exist_ok=True)
    for task in FLAGS.TASK:
        print(f"Generating synthetic data for task: {task}")
        ### special exception for correlate values
        if task in ["correlate_values", "correlate_values_relative"]:
            ### remove shape from encodings if present
            encodings = [
                enc for enc in FLAGS.ENCODING if enc not in ["shape", "color_nominal"]
            ]
            ### if FLAGS.ENCODING is empty raise an error
            if len(FLAGS.ENCODING) < 1:
                raise ValueError("No valid encodings entered for current task")
        else:
            encodings = FLAGS.ENCODING
        task_metadata = generate_charts_and_metadata(
            encodings=encodings,
            task=task,
            num_graphs=FLAGS.NUM_GRAPHS,
            data_dir=target_dir,
            image_dir=image_dir,
            seed=FLAGS.SEED,
        )
        ### write out task specific metadata
        with open(
            os.path.join(target_dir, "synthetic_data", f"{task}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(task_metadata, f, indent=2)


if __name__ == "__main__":
    app.run(main)
