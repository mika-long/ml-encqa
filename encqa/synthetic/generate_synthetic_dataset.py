# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import numpy as np
import pandas as pd

A_ascii = 65
X_ascii = 88


def generate_constrained_data(
    num_categories,
    num_points,
    min_distance,
    rng: np.random.Generator,
    max_attempts=500000,
    **kwargs,
):
    """
    Generate a DataFrame with the following constraints:
    - num_categories unique categories
    - num_points total points
    - min_distance between any two points (to prevent overlap)
    """
    unique_max_min_cats = False
    ignore_max_min_condition = kwargs.get("ignore_max_min_condition", False)
    attempts = 0
    ### ensure that there is only one category with a minimum and maximum number of observations and that each category has at least one observation
    while not unique_max_min_cats:
        categories = [chr(i) for i in range(A_ascii, A_ascii + num_categories)]
        remaining_categories = rng.choice(
            categories, size=num_points - num_categories, replace=True
        ).tolist()
        all_categories = categories + remaining_categories
        rng.shuffle(all_categories)
        category_counts = pd.Series(all_categories).value_counts().to_dict()
        ### get the key or keys with the max value
        num_max_categories = len(
            [
                k
                for k, v in category_counts.items()
                if v == max(category_counts.values())
            ]
        )
        num_min_categories = len(
            [
                k
                for k, v in category_counts.items()
                if v == min(category_counts.values())
            ]
        )
        if num_max_categories == num_min_categories == 1:
            unique_max_min_cats = True
        elif ignore_max_min_condition:
            unique_max_min_cats = True
        else:
            attempts += 1
        if attempts == max_attempts:
            raise ValueError(
                f"Could not generate unique max and min categories after {max_attempts} attempts"
            )

    var1 = np.zeros(num_points)
    var2 = np.zeros(num_points)

    ## ensure that each point as defined by var1 and var2 is at least min_distance away from all other points
    for i in range(num_points):
        attempts = 0
        while attempts < max_attempts:
            var1[i] = rng.normal(loc=50, scale=5)
            var2[i] = rng.normal(loc=50, scale=5)

            if i == 0:
                break

            distances = np.sqrt((var1[i] - var1[:i]) ** 2 + (var2[i] - var2[:i]) ** 2)
            if np.all(distances >= min_distance):
                break

            attempts += 1

        if attempts == max_attempts:
            raise ValueError(
                f"Could not generate point {i+1} within constraints after {max_attempts} attempts"
            )
    df = pd.DataFrame({"cat": all_categories, "var1": var1, "var2": var2})
    ## ensure that the category with the highest and lowest mean are at least 2 units apart
    df["cat"] = pd.Categorical(
        df["cat"], categories=sorted(df["cat"].unique()), ordered=True
    )

    minimum_mean_diff = False
    attempts = 0
    while minimum_mean_diff is False:
        grouped_means = (
            df.groupby("cat", observed=True)["var1"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        ### check that the cat with the higest mean is higher than the second highest by a value of at least 2
        if grouped_means.iloc[0]["var1"] - grouped_means.iloc[1]["var1"] < min_distance:
            df.loc[df["cat"] == grouped_means.iloc[0]["cat"], "var1"] = (
                df.loc[df["cat"] == grouped_means.iloc[0]["cat"], "var1"] * 2
            )
        if (
            grouped_means.iloc[-2]["var1"] - grouped_means.iloc[-1]["var1"]
            < min_distance
        ):
            df.loc[df["cat"] == grouped_means.iloc[-1]["cat"], "var1"] = (
                df.loc[df["cat"] == grouped_means.iloc[-1]["cat"], "var1"] / 2
            )
            attempts += 1
        else:
            minimum_mean_diff = True
        if attempts == max_attempts:
            raise ValueError(
                f"Could not generate differenet enough means after {max_attempts} attempts"
            )

        ### ensure positive values
        if np.any(df["var1"] < 0):
            df["var1"] = df["var1"] + np.abs(df["var1"].min()) + 5
        if np.any(df["var2"] < 0):
            df["var2"] = df["var2"] + np.abs(df["var2"].min()) + 5

    return df


def generate_constrained_outlier_data(
    num_categories,
    num_points,
    min_distance,
    rng: np.random.Generator,
    variable_type,
    outlier_type,
    max_attempts=50000,
):
    """
    Generate a DataFrame with the following constraints:
    - num_categories unique categories
    - num_points total points
    - min_distance between any two points (to prevent overlap)
    - variable_type: "quantitative" or "nominal" is the kind of variable being queried
    - outlier_type: "small" or "large" is the kind of outlier being generated

    """
    categories = [chr(i) for i in range(A_ascii, A_ascii + num_categories)]

    # Randomly choose one category out of the num_categories to be the outlier
    outlier_category = rng.choice(categories)

    # Make a list of all the categories except the outlier
    non_outlier_categories = [cat for cat in categories if cat != outlier_category]
    ## ensure at least 2 of each non outlier category
    if variable_type == "nominal":
        num_non_outlier_categories = 2
        ### randomly choose a base category from the non_outlier_categories
        # base_categories = rng.choice(
        #     non_outlier_categories, size=num_non_outlier_categories, replace=False
        # )
        remaining_category_labels = []
        if outlier_type == "small":
            for non_outlier_category in non_outlier_categories:
                remaining_category_labels += [non_outlier_category] * (
                    (num_points - 1) // num_non_outlier_categories
                )
            if (num_points - 1) % num_non_outlier_categories != 0:
                for i in range((num_points - 1) % num_non_outlier_categories):
                    remaining_category_labels += rng.choice(non_outlier_categories)
            outlier_category_labels = [outlier_category]
        elif outlier_type == "large":
            for non_outlier_category in non_outlier_categories:
                remaining_category_labels += non_outlier_category
            outlier_category_labels = [outlier_category] * (
                num_points - len(remaining_category_labels)
            )
        all_categories = outlier_category_labels + remaining_category_labels
    elif variable_type == "quantitative":
        remaining_category_labels = rng.choice(
            categories, size=num_points - num_categories, replace=True
        ).tolist()

        all_categories = categories + remaining_category_labels

    var1 = np.zeros(num_points)
    var2 = np.zeros(num_points)
    df = pd.DataFrame({"cat": all_categories, "var1": var1, "var2": var2})
    df["cat"] = pd.Categorical(
        df["cat"], categories=sorted(df["cat"].unique()), ordered=True
    )

    # Prevent points in 2d scatter plots from overlapping
    for i in range(num_points):
        attempts = 0
        while attempts < max_attempts:
            df.loc[i, "var1"] = rng.normal(loc=50, scale=5)
            df.loc[i, "var2"] = rng.normal(loc=50, scale=5)
            if i == 0:
                break
            distances = np.sqrt(
                (df.loc[i, "var1"] - df.loc[: i - 1, "var1"]) ** 2
                + (df.loc[i, "var2"] - df.loc[: i - 1, "var2"]) ** 2
            )
            # we only generate 2d scatter plots for nominal type data
            if variable_type != "nominal" or np.all(distances >= min_distance):
                break

            attempts += 1

        if attempts == max_attempts:
            raise ValueError(
                f"Could not generate point {i+1} within constraints after {max_attempts} attempts"
            )
    ### add the outlier point manually for quantitative variables based on the Tukey defn.
    if variable_type == "quantitative":
        if outlier_type == "small":
            df.loc[df["cat"] == outlier_category, "var1"] = df.loc[
                df["cat"] != outlier_category, "var1"
            ].quantile(0.25) - 3 * (
                df.loc[df["cat"] != outlier_category, "var1"].quantile(0.75)
                - df.loc[df["cat"] != outlier_category, "var1"].quantile(0.25)
            )
        elif outlier_type == "large":
            df.loc[df["cat"] == outlier_category, "var1"] = df.loc[
                df["cat"] != outlier_category, "var1"
            ].quantile(0.75) + 3 * (
                df.loc[df["cat"] != outlier_category, "var1"].quantile(0.75)
                - df.loc[df["cat"] != outlier_category, "var1"].quantile(0.25)
            )

        ### ensure positive values
    if np.any(df["var1"] < 0):
        df["var1"] = df["var1"] + np.abs(df["var1"].min()) + 5
    if np.any(df["var2"] < 0):
        df["var2"] = df["var2"] + np.abs(df["var2"].min()) + 5
    return df


def generate_correlated_data(num_points, r, rng: np.random.Generator):
    """
    Generate a DataFrame with the following constraints:
    - num_points total points
    - r is the correlation coefficient between var1 and var2
    """
    var1 = rng.normal(0, 1, num_points)
    var2 = rng.normal(0, 1, num_points)
    data = np.stack((var1, var2), axis=1)

    ## get the covariance matrix with the desired r
    cov = np.array([[1**2, r * 1 * 1], [r * 1 * 1, 1**2]])
    # Apply a Cholesky transformation to the covariance matrix
    L = np.linalg.cholesky(cov)
    data = data @ L.T
    df = pd.DataFrame({"var1": data[:, 0] + 5, "var2": data[:, 1] + 5})

    ### ensure positive values
    if np.any(df["var1"] < 0):
        df["var1"] = df["var1"] + np.abs(df["var1"].min()) + 5
    if np.any(df["var2"] < 0):
        df["var2"] = df["var2"] + np.abs(df["var2"].min()) + 5

    return df


def generate_paired_data(
    num_categories: list[int],
    num_points: int,
    min_distance: float,
    rng: np.random.Generator,
    variable_type: str,
    max_attempts=500000,
    **kwargs,
):
    """
    Generates paired data for 'relative' versions of tasks
    Generate a DataFrame with the following constraints:
    - num_categories unique categories
    - num_points total points
    - min_distance between any two points (to prevent overlap)
    - variable_type: "quantitative" or "nominal" is the kind of variable being
    queried

    """
    # var3_cats = [chr(X_ascii + i) for i in range(len(num_categories))]
    var3_cats = ["LEFT", "RIGHT"]
    rng.shuffle(var3_cats)
    dfs = []
    for g, this_df_num_categories in enumerate(num_categories):
        categories = [chr(i) for i in range(A_ascii, A_ascii + this_df_num_categories)]
        remaining_categories = rng.choice(
            categories, size=num_points - this_df_num_categories, replace=True
        ).tolist()
        all_categories = categories + remaining_categories
        rng.shuffle(all_categories)
        var1 = np.zeros(num_points)
        var2 = np.zeros(num_points)

        ## ensure that each point as defined by var1 and var2 is at least min_distance away from all other points
        for i in range(num_points):
            attempts = 0
            while attempts < max_attempts:
                if variable_type == "nominal":
                    var1[i] = rng.normal(loc=50, scale=5)
                    var2[i] = rng.normal(loc=50, scale=5)
                elif variable_type == "quantitative":
                    var1[i] = rng.normal(loc=50, scale=5)
                    var2[i] = rng.normal(loc=70, scale=5)

                if i == 0 or variable_type == "quantitative":
                    break
                if variable_type == "nominal":
                    distances = np.sqrt(
                        (var1[i] - var1[:i]) ** 2 + (var2[i] - var2[:i]) ** 2
                    )
                    if np.all(distances >= min_distance):
                        break
                attempts += 1

            if attempts == max_attempts:
                raise ValueError(
                    f"Could not generate point {i+1} within constraints after {max_attempts} attempts"
                )
        ###randomly flip the order of the variables
        if rng.integers(2) == 0:
            var1, var2 = var2, var1
        df = pd.DataFrame(
            {
                "cat": all_categories,
                "var1": var1,
                "var2": var2,
                "var3": [var3_cats[g]] * var1.shape[0],
            }
        )
        ## ensure that the category with the highest and lowest mean are at least 2 units apart
        df["cat"] = pd.Categorical(
            df["cat"], categories=sorted(df["cat"].unique()), ordered=True
        )
        dfs.append(df)

    full_df = pd.concat(dfs)

    return full_df


def generate_paired_correlated_data(
    num_points: int, rng: np.random.Generator, **kwargs
):
    """
    Generates paired data for 'relative' versions of correlate values
    Generate a DataFrame with the following constraints:
    - num_points total points
    - r is the correlation coefficient between var1 and var2, but here we set it manually
    """
    high_cor_df = generate_correlated_data(num_points=num_points, r=0.9, rng=rng)
    low_cor_df = generate_correlated_data(num_points=num_points, r=0.1, rng=rng)
    dfs = [high_cor_df, low_cor_df]
    ### shuffle the order of the dfs
    rng.shuffle(dfs)
    full_df = pd.concat(dfs)
    var3 = np.array(["LEFT"] * num_points + ["RIGHT"] * num_points)
    full_df["var3"] = var3

    return full_df
