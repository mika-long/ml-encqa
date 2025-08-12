# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from collections import defaultdict
import ast
import re


def is_valid_literal(s: str):
    """
    check if a string is a valid python literal
    """
    try:
        ast.literal_eval(s)
        return True
    except (ValueError, SyntaxError):
        return False


def extract_answer(row: dict):
    """
    Extract the model response from the model response string using regex pattern matching.
    Also count the number of times each regex pattern is matched.
    If no regex pattern matches, increment the 7th key in pattern_counter.
    """
    text = row["model_response"].strip(". ")
    no_match = True  # Flag to check if no pattern matches

    # Dictionary of regex patterns
    regex_patterns = {
        1: r"^.*?(Answer:|is |is approximately)\s*",  # Leading phrases
        2: r"(\.?\s*\n?Rationale:.*|for.*)$",  # Trailing phrases
        3: r"^[a-zA-Z]+(?:,\s*[a-zA-Z]+)*$",  # Comma-separated words
        4: r"^\[?([a-zA-Z]+(?:,\s*[a-zA-Z]+)*)\]?",  # Comma-separated list
        5: r"\b(one|two|three|four|five|six|seven|eight|nine|zero|ten)\b",  # number words
        6: r"\d+\.(\d*)?",  # numeric digits with optional fractional part
    }
    # Counter dictionary. Can be used for debugging answer extraction.
    pattern_counter = defaultdict(int)

    # Remove leading phrases
    text = re.sub(regex_patterns[1], "", text, flags=re.IGNORECASE)
    if re.search(regex_patterns[1], text, flags=re.IGNORECASE):
        pattern_counter[1] += 1
        no_match = False
    # Remove trailing phrases
    text = re.sub(regex_patterns[2], "", text, flags=re.IGNORECASE)
    if re.search(regex_patterns[2], text, flags=re.IGNORECASE):
        pattern_counter[2] += 1
        no_match = False
    text = text.strip()

    result = None

    if row["answer_type"] == "set":
        if is_valid_literal(text):
            result = ast.literal_eval(text)
        elif re.match(regex_patterns[3], text):
            result = re.split(r",\s*", text)
            pattern_counter[3] += 1
            no_match = False
        elif re.match(regex_patterns[4], text):
            result = re.split(r",\s*", re.match(regex_patterns[4], text).group(1))
            pattern_counter[4] += 1
            no_match = False
        else:
            result = [text]

        result = [
            str(x).strip().lower()
            for x in (result if isinstance(result, list) else [result])
            if x is not None
        ]

    elif row["answer_type"] == "numeric":
        word_to_int = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "zero": "0",
            "ten": "10",
        }
        match_word = re.search(regex_patterns[5], text.lower())
        pattern_counter[5] += 1 if match_word else 0

        match_digit = re.search(regex_patterns[6], text)
        pattern_counter[6] += 1 if match_digit else 0

        if match_word:
            matched_text = match_word.group(1)
            result = float(word_to_int[matched_text])
            no_match = False
        elif match_digit:
            result = float(match_digit.group(0))  # Convert numeric string to float
            no_match = False
        else:
            try:
                result = float(text)
            except ValueError:
                result = np.nan

    elif row["answer_type"] == "multiple_choice":
        result = text if text else ""
    # Increment the 7th counter if no pattern matched
    if no_match:
        pattern_counter[7] += 1

    return result, dict(pattern_counter)


def make_model_response_numeric(row: dict):
    """
    Make the model response a float if the answer type is numeric
    """
    if row["answer_type"] == "numeric":
        try:
            return float(row["model_response_clean"])
        except ValueError:
            return np.nan
    else:
        return row["model_response_clean"]


def compute_match_accuracy(row: dict):
    """
    compute exact match accuracy
    """

    assert row["answer_type"] == "multiple_choice"
    parsed_acc = row["model_response_clean"].lower() == row["true_label"].lower()

    if "lm_eval" in row and (pd.notna(row["lm_eval"]) or row["lm_eval"] == ""):
        return parsed_acc or row["lm_eval"] == "True"
    else:
        return parsed_acc


def compute_abs_error(row: dict):
    """
    compute the absolute difference between model prediction and true label for numeric answers
    """

    assert row["answer_type"] == "numeric"
    model_response = row["model_response_clean"]
    true_label = get_true_label(row)
    return np.abs(model_response - true_label)


def compute_symmetric_abs_error(row: dict):
    """
    compute the absolute difference between model prediction and true label for numeric answers
    """

    assert row["answer_type"] == "numeric"
    model_response = row["model_response_clean"]
    true_label = get_true_label(row)

    numerator = np.abs(model_response - true_label)
    denom = np.abs(true_label) + np.abs(model_response) / 2
    return numerator / denom


def compute_abs_percentage_error(row: dict, eps: float = np.finfo(np.float64).eps):
    """
    compute the absolute percentage error between model prediction and true label for numeric answers
    """

    assert row["answer_type"] == "numeric"
    model_response = row["model_response_clean"]
    true_label = get_true_label(row)
    return np.abs(model_response - true_label) / np.maximum(np.abs(true_label), eps)


def compute_relaxed_accuracy(row: dict):
    """
    compute relaxed accuracy for numeric answers
    """
    assert row["answer_type"] == "numeric"
    try:
        model_response = row["model_response_clean"]
        true_label = get_true_label(row)
        if np.abs(model_response - true_label) <= 0.05 * np.abs(true_label):
            return 1
        else:
            return 0
    except ValueError:
        return np.nan


def compute_exact_set_match(row):
    """
    For set answers, compute the exact set match between the model response and the true label
    """
    assert row["answer_type"] == "set"
    true_label = get_true_label(row)
    model_response = row["model_response_clean"]
    model_response = set([x.lower() for x in model_response])
    parsed_acc = 1 if true_label == model_response else 0

    if "lm_eval" in row and (pd.notna(row["lm_eval"]) or row["lm_eval"] == ""):
        return parsed_acc or row["lm_eval"] == "True"
    else:
        return parsed_acc


def compute_jaccard_similarity(row):
    """
    for set answers, compute the jaccard similarity between the model response and the true label
    """
    assert row["answer_type"] == "set"
    true_label = get_true_label(row)
    model_response = row["model_response_clean"]

    intersection = len(true_label.intersection(set(model_response)))
    union = len(true_label.union(set(model_response)))
    js = intersection / union if union != 0 else 0
    return js


def compute_overlap_similarity(row):
    """
    computes the overlap or Szymkiewicz–Simpson coefficient
    read more here - https://en.wikipedia.org/wiki/Overlap_coefficient
    """
    assert row["answer_type"] == "set"
    true_label = get_true_label(row)
    model_response = row["model_response_clean"]
    if model_response is None:
        model_response = []
    if len(set(true_label)) < 1 and len(set(model_response)) < 1:
        return 0
    else:
        oc = len(set(true_label).intersection(set(model_response))) / np.minimum(
            len(set(true_label)), len(set(model_response))
        )
        return oc


def get_true_label(row: dict):
    """
    get the true label from the row in the appropriate datatype
    """
    if row["answer_type"] == "numeric":
        try:
            return float(row["true_label"])
        except ValueError:
            print(f"Error converting {row} to float")
    elif row["answer_type"] == "multiple_choice":
        return row["true_label"]
    elif row["answer_type"] == "set":
        return set([x.lower() for x in ast.literal_eval(row["true_label"])])
    else:
        raise ValueError(f"Answer type {row['answer_type']} not recognized")


def compute_scaled_abs_error(true_labels: list, model_responses: list):
    """
    creates a minmax scaler based on true labels and applies it to the model responses before computing the absolute error
    """
    true_labels = np.array(true_labels, dtype=float).reshape(-1, 1)
    model_responses = np.array(model_responses, dtype=float).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(true_labels)
    true_labels = scaler.transform(true_labels)
    model_responses = scaler.transform(model_responses)

    return np.abs(model_responses - true_labels)


def compute_accuracy(row: dict):
    """
    compute the accuracy based on the answer type
    """
    if row["answer_type"] == "numeric":
        return compute_relaxed_accuracy(row)
    elif row["answer_type"] == "multiple_choice":
        return compute_match_accuracy(row)
    elif row["answer_type"] == "set":
        return compute_exact_set_match(row)
    else:
        return None


def add_metrics(df: pd.DataFrame):
    """
    top level function that adds all the metrics to the dataframe
    This assumues the dataframe has the fields present in the dataset
    as well as a "model_response" column
    """
    ## get masks for each answer type
    numeric_rows = df["answer_type"] == "numeric"
    set_rows = df["answer_type"] == "set"
    # df["model_response_clean"] = df.apply(extract_answer, axis=1)
    result, counts = zip(*df.apply(extract_answer, axis=1))
    df["model_response_clean"] = result
    df["pattern_counts"] = counts
    df["accuracy"] = df.apply(compute_accuracy, axis=1)
    ### initialize each metric as a column of NaNs
    df["absolute_error"] = np.nan
    df["absolute_percentage_error"] = np.nan
    df["scaled_absolute_error"] = np.nan
    df["overlap_similarity"] = np.nan
    df["jaccard_similarity"] = np.nan
    ### compute the metrics for each answer type
    df.loc[numeric_rows, "absolute_error"] = df.loc[numeric_rows].apply(
        compute_abs_error, axis=1
    )
    df.loc[numeric_rows, "absolute_percentage_error"] = df.loc[numeric_rows].apply(
        compute_abs_percentage_error, axis=1
    )
    df.loc[numeric_rows, "symmetric_absolute_percentage_error"] = df.loc[
        numeric_rows
    ].apply(compute_symmetric_abs_error, axis=1)

    df.loc[set_rows, "overlap_similarity"] = df.loc[set_rows].apply(
        compute_overlap_similarity, axis=1
    )
    df.loc[set_rows, "jaccard_similarity"] = df.loc[set_rows].apply(
        compute_jaccard_similarity, axis=1
    )
    numeric_rows = df["answer_type"] == "numeric"
    df.loc[numeric_rows, "scaled_absolute_error"] = compute_scaled_abs_error(
        df.loc[numeric_rows, "true_label"], df.loc[numeric_rows, "model_response_clean"]
    ).flatten()

    return df
