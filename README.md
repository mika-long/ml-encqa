# EncQA: Benchmarking Vision-Language Models on Visual Encodings for Charts

This repository accompanies the research paper: [EncQA: Benchmarking Vision-Language Models on Visual Encodings for Charts](https://arxiv.org/abs/2508.04650v1) ([bibtex](#Cite)).

## Abstract

Multimodal vision-language models (VLMs) continue to show ever-improving scores on chart understanding benchmarks but do these improvements truly capture improvements in models’ visual reasoning capabilities? We introduce EncQA, a novel benchmark designed to provide systematic coverage of fundamental visual encodings and chart understanding tasks that are crucial for chart comprehension. EncQA provides 2,076 synthetic charts paired with 2,250 questions that span six visual encoding channels (position, length, area, color quantitative, color nominal, and shape) and six task types (find extrema, retrieve value, find anomaly, filter values, compute derived value, and correlate values). Our evaluation of 11 state-of-the-art VLMs — including proprietary and open-source models — reveals that performance varies significantly across different encoding-task combinations and does not necessarily improve with model size. Our results suggest that current approaches to improving chart understanding capabilities may benefit from more targeted strategies that address specific visual reasoning gaps beyond simply increasing model and dataset size.

## Benchmark Data

We also provide a [dataset loader here](encqa/enc_qa_dataset.py). This [test file](encqa/enc_qa_dataset_test.py) describes how to load it. You can run the test with this command (assuming you have [uv installed](https://docs.astral.sh/uv/).)

```python
uv run pytest -s
```

## Data Format

The data loader will produce a list of dictionaries with the following format.

```py
{
    "image": PILImage,
    "image_path": string,       # path to image if you have generated a local dataset.
    "question": string,
    "true_label": string,       
    "options": list[string],    # valid options for multiple choice or set questions
    "task": string,             
    "task_details": string,     # JSON object with extra task details (e.g. minimum or maximum for find_extrema)
    "encoding": string,         
    "variable_type": string,    # quantitative|nominal
    "answer_type": string,      # set|numeric|multiple_choice
    "num_marks": number,
    "num_categories": number, 
    "chart_spec": string,       # The vega-lite spec for the chart
    "split": string,            # We currently have a single split
    "canary_guid": string,      # We add a canary string to enable detection of inadvertent data leakage in future.
}
```

## Eval Helpers

We provide helper functions for computing accuracy for model responses in `encqa/eval_metrics.py`. The main utility function is `add_metrics` which takes a pandas dataframe that has the benchmark data and an additional `model_responses` column. Note that you may have to adjust the `extract_answer` function to handle the output of your model correctly.

## Generating new data

If you want to generate new data we provide the generate_synthetic.py script. Which can be run in the following way:

```python
uv run -m encqa.generate_synthetic.py --TARGET_DIR="path/to/output_folder/" --SEED=1234
```

See the script for more options that you can pass into it. For example you can generate data for only a subset of task-encoding pairs, or generate more or fewer charts.

## Cite

```
@misc{mukherjee2025encqabenchmarkingvisionlanguagemodels,
      title={EncQA: Benchmarking Vision-Language Models on Visual Encodings for Charts}, 
      author={Kushin Mukherjee and Donghao Ren and Dominik Moritz and Yannick Assogba},
      year={2025},
      eprint={2508.04650},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.04650}, 
}
```