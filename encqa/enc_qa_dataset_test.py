# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""Test Datasets Module"""

from encqa import enc_qa_dataset


def test_load_hf_dataset():
    encqa_data = enc_qa_dataset.load_enc_qa_hf(dataset_path="data/hf")
    print(encqa_data[0])
    assert len(encqa_data) == 2525


def test_load_local_dataset():
    # Replace data/ with a path to a valid encqa data dir
    encqa_wrapper = enc_qa_dataset.load_enc_qa(data_dir="data/")
    print("Loaded data from local dir")
    encqa_data: list[dict] = encqa_wrapper.data()
    print(encqa_data[0])
    assert len(encqa_data) == 2525
