# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""A script for creating a HF dataset for encqa"""

from absl import app, flags  # type: ignore
import os
import enc_qa_dataset

"""
Usage: python enc_qa/convert_to_dataset.py --TASK=task_name  --NUM_GRAPHS=num_graphs
"""
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "SOURCE_DIR",
    "data/",
    "directory to save the data to",
)


flags.DEFINE_string(
    "OUTPUT_DIR",
    "data/hf",
    "directory to save the data to",
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


def main(argv):
    del argv

    SOURCE_DIR = FLAGS.SOURCE_DIR
    OUTPUT_DIR = FLAGS.OUTPUT_DIR
    TASKS = FLAGS.TASK

    # Load encqa
    encqa = enc_qa_dataset.load_enc_qa(data_dir=SOURCE_DIR, tasks=TASKS)
    encqa_hf = encqa.to_hf_dataset()

    encqa_hf.save_to_disk(OUTPUT_DIR)
    encqa_hf.to_pandas().to_parquet(os.path.join("encqa_v1.parquet"), index=False)


if __name__ == "__main__":
    app.run(main)
