# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.


def return_encoding_answer_dict(encoding: str, **kwargs):
    """
    converts answer keys to color or shape for nominal encoding
    encoding:str
    color or shape"""
    vega_lite_spec = kwargs.get("vega_lite_spec", None)
    if encoding == "color_nominal":
        if vega_lite_spec:
            return {
                "A": "#1f77b4",
                "B": "#ff7f0e",
                "C": "#2ca02c",
                "D": "#d62728",
                "E": "#9467bd",
                "F": "#8c564b",
            }

        else:
            return {
                "A": "Blue",
                "B": "Orange",
                "C": "Green",
                "D": "Red",
                "E": "Purple",
                "F": "Brown",
            }
    if encoding == "shape":
        if vega_lite_spec:
            return {
                "A": "circle",
                "B": "square",
                "C": "triangle",
                "D": "cross",
                "E": "M0,.5L.6,.8L.5,.1L1,-.3L.3,-.4L0,-1L-.3,-.4L-1,-.3L-.5,.1L-.6,.8L0,.5Z",
                "F": "diamond",
            }
        else:
            return {
                "A": "circle",
                "B": "square",
                "C": "triangle",
                "D": "cross",
                "E": "star",
                "F": "diamond",
            }

    else:
        return ValueError("encoding must be either 'color' or 'shape'")


def get_encoding_variable_types(encoding: str):
    """
    returns the variable type for a given encoding
    encoding:str
    color or shape"""
    if encoding in ["position", "area", "length", "color_quantitative"]:
        return "quantitative"
    elif encoding in ["shape", "color_nominal"]:
        return "nominal"
    else:
        return ValueError("encoding must be either 'color' or 'shape'")
