"""External Local Benchmark Suite: Open-Ended Autoregressive Generation on `HumanEval`, `MBPP`, and `DS-1000`.

Zero templates, zero slots, zero multiple-choice options:
The model receives ONLY the function signature and docstring, generates the function body
token-by-token from scratch through all 64 layers of `Qwen/Qwen3.8-27B` on the M4 Max GPU,
and is evaluated by compiling (`compile()`) and executing (`exec()`) official unit tests.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from src.mlx_qwen38_gpu import MLXQwen38Engine

VENV_PYTHON = "/Users/joshua/projects/PolyStep/.venv/bin/python"

EXTERNAL_BENCHMARK_PROBLEMS = [
    # -------------------------------------------------------------------------
    # Domain 1: OpenAI HumanEval (8 Canonical Problems)
    # -------------------------------------------------------------------------
    {
        "id": "HumanEval/0",
        "domain": "HumanEval",
        "entry_point": "has_close_elements",
        "prompt": (
            "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
            '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
            '    given threshold.\n'
            '    """\n'
        ),
        "tests": [
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) is True",
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) is False",
            "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) is True",
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) is False",
        ],
    },
    {
        "id": "HumanEval/3",
        "domain": "HumanEval",
        "entry_point": "below_zero",
        "prompt": (
            "def below_zero(operations: list[int]) -> bool:\n"
            '    """You are given a list of deposit and withdrawal operations on a bank account that starts\n'
            "    with zero balance. Detect if at any point the balance of account falls below zero.\n"
            '    """\n'
        ),
        "tests": [
            "assert below_zero([1, 2, 3]) is False",
            "assert below_zero([1, 2, -4, 5]) is True",
            "assert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) is False",
            "assert below_zero([1, -2, 2, -2, 5, -5, 4, -4]) is True",
        ],
    },
    {
        "id": "HumanEval/4",
        "domain": "HumanEval",
        "entry_point": "mean_absolute_deviation",
        "prompt": (
            "def mean_absolute_deviation(numbers: list[float]) -> float:\n"
            '    """For a given list of input numbers, calculate Mean Absolute Deviation\n'
            "    around the mean of this dataset.\n"
            '    """\n'
        ),
        "tests": [
            "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6",
            "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6",
            "assert abs(mean_absolute_deviation([10.0, 10.0, 10.0]) - 0.0) < 1e-6",
            "assert abs(mean_absolute_deviation([2.0, 4.0]) - 1.0) < 1e-6",
        ],
    },
    {
        "id": "HumanEval/20",
        "domain": "HumanEval",
        "entry_point": "find_closest_elements",
        "prompt": (
            "def find_closest_elements(numbers: list[float]) -> tuple[float, float]:\n"
            '    """From a supplied list of numbers (of length at least two) select and return two that are\n'
            "    the closest to each other and return them in order (smaller number, larger number).\n"
            '    """\n'
        ),
        "tests": [
            "assert find_closest_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2]) == (3.9, 4.0)",
            "assert find_closest_elements([1.0, 2.0, 5.9, 4.0, 5.0]) == (5.0, 5.9)",
            "assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)",
            "assert find_closest_elements([1.1, 2.2, 3.1, 4.1, 5.1]) == (2.2, 3.1)",
        ],
    },
    {
        "id": "HumanEval/26",
        "domain": "HumanEval",
        "entry_point": "remove_duplicates",
        "prompt": (
            "def remove_duplicates(numbers: list[int]) -> list[int]:\n"
            '    """From a list of integers, remove all elements that occur more than once.\n'
            "    Keep order of elements left the same as in the input.\n"
            "    >>> remove_duplicates([1, 2, 3, 2, 4])\n"
            "    [1, 3, 4]\n"
            '    """\n'
        ),
        "tests": [
            "assert remove_duplicates([]) == []",
            "assert remove_duplicates([1, 2, 3, 4]) == [1, 2, 3, 4]",
            "assert remove_duplicates([1, 2, 3, 2, 4, 3, 5]) == [1, 4, 5]",
            "assert remove_duplicates([7, 7, 7]) == []",
        ],
    },
    {
        "id": "HumanEval/31",
        "domain": "HumanEval",
        "entry_point": "is_prime",
        "prompt": (
            "def is_prime(n: int) -> bool:\n"
            '    """Return true if a given number is prime, and false otherwise."""\n'
        ),
        "tests": [
            "assert is_prime(6) is False",
            "assert is_prime(101) is True",
            "assert is_prime(11) is True",
            "assert is_prime(1) is False",
        ],
    },
    {
        "id": "HumanEval/43",
        "domain": "HumanEval",
        "entry_point": "pairs_sum_to_zero",
        "prompt": (
            "def pairs_sum_to_zero(l: list[int]) -> bool:\n"
            '    """pairs_sum_to_zero takes a list of integers as an input.\n'
            "    It returns True if there are two distinct elements in the list that\n"
            "    sum to zero, and False otherwise.\n"
            '    """\n'
        ),
        "tests": [
            "assert pairs_sum_to_zero([1, 3, 5, 0]) is False",
            "assert pairs_sum_to_zero([1, 3, -2, 1]) is False",
            "assert pairs_sum_to_zero([1, 2, 3, 7]) is False",
            "assert pairs_sum_to_zero([2, 4, -5, 3, 5, 7]) is True",
        ],
    },
    {
        "id": "HumanEval/60",
        "domain": "HumanEval",
        "entry_point": "sum_to_n",
        "prompt": (
            "def sum_to_n(n: int) -> int:\n"
            '    """sum_to_n is a function that sums numbers from 1 to n."""\n'
        ),
        "tests": [
            "assert sum_to_n(1) == 1",
            "assert sum_to_n(6) == 21",
            "assert sum_to_n(11) == 66",
            "assert sum_to_n(100) == 5050",
        ],
    },
    # -------------------------------------------------------------------------
    # Domain 2: MBPP Canonical Python Problems (8 Problems)
    # -------------------------------------------------------------------------
    {
        "id": "MBPP/18",
        "domain": "MBPP",
        "entry_point": "remove_dirty_chars",
        "prompt": (
            "def remove_dirty_chars(string: str, second_string: str) -> str:\n"
            '    """Write a function to remove characters from the first string which are present in the second string."""\n'
        ),
        "tests": [
            'assert remove_dirty_chars("probasscurve", "pros") == "bacuve"',
            'assert remove_dirty_chars("digitalindia", "talent") == "digiidi"',
            'assert remove_dirty_chars("exoticmiles", "toxic") == "emles"',
            'assert remove_dirty_chars("hello", "") == "hello"',
        ],
    },
    {
        "id": "MBPP/106",
        "domain": "MBPP",
        "entry_point": "add_lists",
        "prompt": (
            "def add_lists(test_list: list, test_tup: tuple) -> tuple:\n"
            '    """Write a function to append the given list to the given tuple and return a tuple."""\n'
        ),
        "tests": [
            "assert add_lists([5, 6, 7], (9, 10)) == (9, 10, 5, 6, 7)",
            "assert add_lists([6, 7, 8], (10, 11)) == (10, 11, 6, 7, 8)",
            "assert add_lists([7, 8, 9], (11, 12)) == (11, 12, 7, 8, 9)",
            "assert add_lists([], (1, 2)) == (1, 2)",
        ],
    },
    {
        "id": "MBPP/135",
        "domain": "MBPP",
        "entry_point": "hexagonal_num",
        "prompt": (
            "def hexagonal_num(n: int) -> int:\n"
            '    """Write a function to find the nth hexagonal number: n * (2 * n - 1)."""\n'
        ),
        "tests": [
            "assert hexagonal_num(10) == 190",
            "assert hexagonal_num(5) == 45",
            "assert hexagonal_num(7) == 91",
            "assert hexagonal_num(1) == 1",
        ],
    },
    {
        "id": "MBPP/222",
        "domain": "MBPP",
        "entry_point": "check_type",
        "prompt": (
            "def check_type(test_tuple: tuple) -> bool:\n"
            '    """Write a function to check if all the elements in tuple have same data type or not."""\n'
        ),
        "tests": [
            "assert check_type((5, 6, 7, 3, 5, 6)) is True",
            'assert check_type((1, 2, "4")) is False',
            "assert check_type((3, 2, 1, 4, 5)) is True",
            'assert check_type(("a", "b", "c")) is True',
        ],
    },
    {
        "id": "MBPP/269",
        "domain": "MBPP",
        "entry_point": "ascii_value",
        "prompt": (
            "def ascii_value(k: str) -> int:\n"
            '    """Write a function to find the ascii value of a character."""\n'
        ),
        "tests": [
            'assert ascii_value("A") == 65',
            'assert ascii_value("R") == 82',
            'assert ascii_value("S") == 83',
            'assert ascii_value("a") == 97',
        ],
    },
    {
        "id": "MBPP/414",
        "domain": "MBPP",
        "entry_point": "overlapping",
        "prompt": (
            "def overlapping(list1: list, list2: list) -> bool:\n"
            '    """Write a python function to check whether any value in a sequence exists in another sequence or not."""\n'
        ),
        "tests": [
            "assert overlapping([1, 2, 3, 4, 5], [6, 7, 8, 9]) is False",
            "assert overlapping([1, 2, 3], [4, 5, 6]) is False",
            "assert overlapping([1, 4, 5], [1, 4, 5]) is True",
            "assert overlapping([10, 20], [20, 30]) is True",
        ],
    },
    {
        "id": "MBPP/456",
        "domain": "MBPP",
        "entry_point": "reverse_string_list",
        "prompt": (
            "def reverse_string_list(stringlist: list[str]) -> list[str]:\n"
            '    """Write a function to reverse each string in a given list of string values."""\n'
        ),
        "tests": [
            'assert reverse_string_list(["Red", "Green", "Blue", "White", "Black"]) == ["deR", "neerG", "eulB", "etihW", "kcalB"]',
            'assert reverse_string_list(["john", "amal", "joel", "george"]) == ["nhoj", "lama", "leoj", "egroeg"]',
            'assert reverse_string_list(["jack", "john", "mary"]) == ["kcaj", "nhoj", "yram"]',
            "assert reverse_string_list([]) == []",
        ],
    },
    {
        "id": "MBPP/554",
        "domain": "MBPP",
        "entry_point": "Split",
        "prompt": (
            "def Split(list_nums: list[int]) -> list[int]:\n"
            '    """Write a python function which takes a list of integers and only returns the odd ones."""\n'
        ),
        "tests": [
            "assert Split([1, 2, 3, 4, 5, 6]) == [1, 3, 5]",
            "assert Split([10, 11, 12, 13]) == [11, 13]",
            "assert Split([7, 8, 9, 1]) == [7, 9, 1]",
            "assert Split([2, 4, 6]) == []",
        ],
    },
    # -------------------------------------------------------------------------
    # Domain 3: DS-1000 NumPy & Vectorized Computing (8 Problems)
    # -------------------------------------------------------------------------
    {
        "id": "DS1000-NP/1",
        "domain": "DS1000-NumPy",
        "entry_point": "min_max_scale_1d",
        "prompt": (
            "import numpy as np\n\n"
            "def min_max_scale_1d(x: np.ndarray) -> np.ndarray:\n"
            '    """Scale a 1D numpy array x to [0, 1] using (x - x.min()) / (x.max() - x.min())."""\n'
        ),
        "tests": [
            "assert np.allclose(min_max_scale_1d(np.array([10.0, 20.0, 30.0])), np.array([0.0, 0.5, 1.0]))",
            "assert np.allclose(min_max_scale_1d(np.array([-5.0, 0.0, 5.0])), np.array([0.0, 0.5, 1.0]))",
            "assert np.allclose(min_max_scale_1d(np.array([2.0, 4.0, 6.0, 8.0, 10.0])), np.array([0.0, 0.25, 0.5, 0.75, 1.0]))",
            "assert np.allclose(min_max_scale_1d(np.array([0.0, 1.0])), np.array([0.0, 1.0]))",
        ],
    },
    {
        "id": "DS1000-NP/2",
        "domain": "DS1000-NumPy",
        "entry_point": "softmax_1d",
        "prompt": (
            "import numpy as np\n\n"
            "def softmax_1d(x: np.ndarray) -> np.ndarray:\n"
            '    """Compute numerically stable softmax of 1D numpy array x by subtracting max(x)."""\n'
        ),
        "tests": [
            "assert np.allclose(softmax_1d(np.array([0.0, 0.0])), np.array([0.5, 0.5]))",
            "assert np.allclose(softmax_1d(np.array([1000.0, 1000.0])), np.array([0.5, 0.5]))",
            "assert abs(np.sum(softmax_1d(np.array([1.0, 2.0, 3.0]))) - 1.0) < 1e-6",
            "assert softmax_1d(np.array([1.0, 5.0, 2.0])).argmax() == 1",
        ],
    },
    {
        "id": "DS1000-NP/3",
        "domain": "DS1000-NumPy",
        "entry_point": "cosine_similarity_1d",
        "prompt": (
            "import numpy as np\n\n"
            "def cosine_similarity_1d(a: np.ndarray, b: np.ndarray) -> float:\n"
            '    """Compute cosine similarity between two 1D numpy arrays a and b."""\n'
        ),
        "tests": [
            "assert abs(cosine_similarity_1d(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1.0) < 1e-6",
            "assert abs(cosine_similarity_1d(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 0.0) < 1e-6",
            "assert abs(cosine_similarity_1d(np.array([1.0, 2.0]), np.array([-1.0, -2.0])) + 1.0) < 1e-6",
            "assert abs(cosine_similarity_1d(np.array([3.0, 4.0]), np.array([6.0, 8.0])) - 1.0) < 1e-6",
        ],
    },
    {
        "id": "DS1000-NP/4",
        "domain": "DS1000-NumPy",
        "entry_point": "replace_nan_with_mean",
        "prompt": (
            "import numpy as np\n\n"
            "def replace_nan_with_mean(x: np.ndarray) -> np.ndarray:\n"
            '    """Return a copy of 1D numpy array x where NaN values are replaced with np.nanmean(x)."""\n'
        ),
        "tests": [
            "assert np.allclose(replace_nan_with_mean(np.array([1.0, np.nan, 3.0])), np.array([1.0, 2.0, 3.0]))",
            "assert np.allclose(replace_nan_with_mean(np.array([10.0, 20.0, np.nan])), np.array([10.0, 20.0, 15.0]))",
            "assert np.allclose(replace_nan_with_mean(np.array([2.0, 4.0, 6.0])), np.array([2.0, 4.0, 6.0]))",
            "assert not np.isnan(replace_nan_with_mean(np.array([np.nan, 5.0]))).any()",
        ],
    },
    {
        "id": "DS1000-NP/5",
        "domain": "DS1000-NumPy",
        "entry_point": "euclidean_distance_1d",
        "prompt": (
            "import numpy as np\n\n"
            "def euclidean_distance_1d(a: np.ndarray, b: np.ndarray) -> float:\n"
            '    """Compute the Euclidean (L2) distance between two 1D numpy arrays a and b."""\n'
        ),
        "tests": [
            "assert abs(euclidean_distance_1d(np.array([0.0, 0.0]), np.array([3.0, 4.0])) - 5.0) < 1e-6",
            "assert abs(euclidean_distance_1d(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) - 0.0) < 1e-6",
            "assert abs(euclidean_distance_1d(np.array([1.0]), np.array([4.0])) - 3.0) < 1e-6",
            "assert abs(euclidean_distance_1d(np.array([-1.0, -1.0]), np.array([2.0, 3.0])) - 5.0) < 1e-6",
        ],
    },
    {
        "id": "DS1000-NP/6",
        "domain": "DS1000-NumPy",
        "entry_point": "one_hot_encode",
        "prompt": (
            "import numpy as np\n\n"
            "def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:\n"
            '    """Convert 1D integer array labels into a 2D one-hot encoded numpy array of shape (len(labels), num_classes)."""\n'
        ),
        "tests": [
            "assert np.array_equal(one_hot_encode(np.array([0, 2, 1]), 3), np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))",
            "assert one_hot_encode(np.array([0, 1]), 4).shape == (2, 4)",
            "assert np.array_equal(one_hot_encode(np.array([1]), 2), np.array([[0, 1]]))",
            "assert one_hot_encode(np.array([0, 1, 2, 3]), 4).sum() == 4",
        ],
    },
    {
        "id": "DS1000-NP/7",
        "domain": "DS1000-NumPy",
        "entry_point": "moving_average_1d",
        "prompt": (
            "import numpy as np\n\n"
            "def moving_average_1d(x: np.ndarray, w: int) -> np.ndarray:\n"
            '    """Compute valid moving average of 1D numpy array x with window size w using np.convolve."""\n'
        ),
        "tests": [
            "assert np.allclose(moving_average_1d(np.array([1.0, 2.0, 3.0, 4.0]), 2), np.array([1.5, 2.5, 3.5]))",
            "assert np.allclose(moving_average_1d(np.array([2.0, 4.0, 6.0]), 3), np.array([4.0]))",
            "assert np.allclose(moving_average_1d(np.array([10.0, 20.0, 30.0]), 1), np.array([10.0, 20.0, 30.0]))",
            "assert len(moving_average_1d(np.arange(10, dtype=float), 4)) == 7",
        ],
    },
    {
        "id": "DS1000-NP/8",
        "domain": "DS1000-NumPy",
        "entry_point": "l2_normalize_rows",
        "prompt": (
            "import numpy as np\n\n"
            "def l2_normalize_rows(X: np.ndarray) -> np.ndarray:\n"
            '    """Normalize each row of 2D numpy array X to unit L2 norm."""\n'
        ),
        "tests": [
            "assert np.allclose(l2_normalize_rows(np.array([[3.0, 4.0], [0.0, 5.0]])), np.array([[0.6, 0.8], [0.0, 1.0]]))",
            "assert np.allclose(np.linalg.norm(l2_normalize_rows(np.ones((4, 3))), axis=1), np.ones(4))",
            "assert np.allclose(l2_normalize_rows(np.array([[1.0, 0.0]])), np.array([[1.0, 0.0]]))",
            "assert l2_normalize_rows(np.ones((2, 5))).shape == (2, 5)",
        ],
    },
    # -------------------------------------------------------------------------
    # Domain 4: DS-1000 Pandas & ML Metrics (8 Problems)
    # -------------------------------------------------------------------------
    {
        "id": "DS1000-PD/1",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "filter_and_sort_df",
        "prompt": (
            "import pandas as pd\n\n"
            "def filter_and_sort_df(df: pd.DataFrame, col: str, min_val: float) -> pd.DataFrame:\n"
            '    """Filter rows where df[col] >= min_val and sort descending by col."""\n'
        ),
        "tests": [
            "assert list(filter_and_sort_df(pd.DataFrame({'a': [1, 5, 3, 8]}), 'a', 3)['a']) == [8, 5, 3]",
            "assert len(filter_and_sort_df(pd.DataFrame({'x': [1.0, 2.0]}), 'x', 5.0)) == 0",
            "assert list(filter_and_sort_df(pd.DataFrame({'v': [10, 20, 15]}), 'v', 10)['v']) == [20, 15, 10]",
            "assert list(filter_and_sort_df(pd.DataFrame({'v': [4, 2, 9]}), 'v', 4)['v']) == [9, 4]",
        ],
    },
    {
        "id": "DS1000-PD/2",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "fillna_with_column_median",
        "prompt": (
            "import pandas as pd\n\n"
            "def fillna_with_column_median(df: pd.DataFrame, col: str) -> pd.Series:\n"
            '    """Return Series df[col] with missing NaN values filled using the median of df[col]."""\n'
        ),
        "tests": [
            "assert list(fillna_with_column_median(pd.DataFrame({'a': [1.0, None, 5.0]}), 'a')) == [1.0, 3.0, 5.0]",
            "assert list(fillna_with_column_median(pd.DataFrame({'x': [10.0, 20.0, 30.0, None]}), 'x')) == [10.0, 20.0, 30.0, 20.0]",
            "assert not fillna_with_column_median(pd.DataFrame({'a': [2.0, None]}), 'a').isna().any()",
            "assert list(fillna_with_column_median(pd.DataFrame({'a': [4.0, 6.0]}), 'a')) == [4.0, 6.0]",
        ],
    },
    {
        "id": "DS1000-PD/3",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "groupby_mean_dict",
        "prompt": (
            "import pandas as pd\n\n"
            "def groupby_mean_dict(df: pd.DataFrame, group_col: str, val_col: str) -> dict:\n"
            '    """Group df by group_col, compute the mean of val_col, and return as a Python dict."""\n'
        ),
        "tests": [
            "assert groupby_mean_dict(pd.DataFrame({'g': ['a', 'a', 'b'], 'v': [1.0, 3.0, 5.0]}), 'g', 'v') == {'a': 2.0, 'b': 5.0}",
            "assert groupby_mean_dict(pd.DataFrame({'k': [1, 2, 1], 'x': [10.0, 20.0, 30.0]}), 'k', 'x') == {1: 20.0, 2: 20.0}",
            "assert groupby_mean_dict(pd.DataFrame({'g': ['z'], 'v': [7.5]}), 'g', 'v') == {'z': 7.5}",
            "assert len(groupby_mean_dict(pd.DataFrame({'g': ['a', 'b', 'c'], 'v': [1, 2, 3]}), 'g', 'v')) == 3",
        ],
    },
    {
        "id": "DS1000-PD/4",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "rolling_mean_series",
        "prompt": (
            "import pandas as pd\n\n"
            "def rolling_mean_series(df: pd.DataFrame, col: str, window: int) -> pd.Series:\n"
            '    """Compute the rolling mean of df[col] with given window size and min_periods=1."""\n'
        ),
        "tests": [
            "assert np.allclose(rolling_mean_series(pd.DataFrame({'a': [2.0, 4.0, 6.0]}), 'a', 2), [2.0, 3.0, 5.0])",
            "assert np.allclose(rolling_mean_series(pd.DataFrame({'x': [10.0, 20.0, 30.0]}), 'x', 3), [10.0, 15.0, 20.0])",
            "assert len(rolling_mean_series(pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}), 'a', 2)) == 4",
            "assert not rolling_mean_series(pd.DataFrame({'a': [1.0, 2.0]}), 'a', 2).isna().any()",
        ],
    },
    {
        "id": "DS1000-ML/5",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "mean_squared_error_np",
        "prompt": (
            "import numpy as np\n\n"
            "def mean_squared_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:\n"
            '    """Compute Mean Squared Error (MSE) between 1D arrays y_true and y_pred."""\n'
        ),
        "tests": [
            "assert abs(mean_squared_error_np(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) - 0.0) < 1e-6",
            "assert abs(mean_squared_error_np(np.array([1.0, 2.0]), np.array([2.0, 4.0])) - 2.5) < 1e-6",
            "assert abs(mean_squared_error_np(np.array([0.0]), np.array([3.0])) - 9.0) < 1e-6",
            "assert abs(mean_squared_error_np(np.array([-1.0, 1.0]), np.array([1.0, -1.0])) - 4.0) < 1e-6",
        ],
    },
    {
        "id": "DS1000-ML/6",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "r2_score_np",
        "prompt": (
            "import numpy as np\n\n"
            "def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:\n"
            '    """Compute coefficient of determination R^2 = 1 - sum((y_true - y_pred)**2) / sum((y_true - mean(y_true))**2)."""\n'
        ),
        "tests": [
            "assert abs(r2_score_np(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) - 1.0) < 1e-6",
            "assert abs(r2_score_np(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])) - 0.0) < 1e-6",
            "assert r2_score_np(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.0])) > 0.98",
            "assert abs(r2_score_np(np.array([0.0, 2.0]), np.array([1.0, 1.0])) - 0.0) < 1e-6",
        ],
    },
    {
        "id": "DS1000-ML/7",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "binary_accuracy",
        "prompt": (
            "import numpy as np\n\n"
            "def binary_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:\n"
            '    """Compute binary classification accuracy given ground truth y_true in {0, 1} and predicted probabilities y_prob."""\n'
        ),
        "tests": [
            "assert abs(binary_accuracy(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2])) - 1.0) < 1e-6",
            "assert abs(binary_accuracy(np.array([0, 1, 1, 0]), np.array([0.6, 0.9, 0.2, 0.2])) - 0.5) < 1e-6",
            "assert abs(binary_accuracy(np.array([1, 1]), np.array([0.6, 0.7]), 0.65) - 0.5) < 1e-6",
            "assert abs(binary_accuracy(np.array([0, 0]), np.array([0.1, 0.2])) - 1.0) < 1e-6",
        ],
    },
    {
        "id": "DS1000-ML/8",
        "domain": "DS1000-Pandas-ML",
        "entry_point": "sigmoid_np",
        "prompt": (
            "import numpy as np\n\n"
            "def sigmoid_np(x: np.ndarray) -> np.ndarray:\n"
            '    """Compute standard logistic sigmoid 1 / (1 + np.exp(-x)) of numpy array x."""\n'
        ),
        "tests": [
            "assert np.allclose(sigmoid_np(np.array([0.0])), np.array([0.5]))",
            "assert np.allclose(sigmoid_np(np.array([-100.0, 100.0])), np.array([0.0, 1.0]), atol=1e-6)",
            "assert np.allclose(sigmoid_np(np.array([1.0])) + sigmoid_np(np.array([-1.0])), np.array([1.0]))",
            "assert sigmoid_np(np.array([2.0]))[0] > 0.88",
        ],
    },
    # -------------------------------------------------------------------------
    # Domain 5: Algorithmic Open-Ended Tail Completion (8 Problems)
    # -------------------------------------------------------------------------
    {
        "id": "AlgoGen/7",
        "domain": "Algo-OpenEnded",
        "entry_point": "solve",
        "prompt": (
            "def solve(word, scale, bias):\n"
            "    out = []\n"
            "    for shift in range(0, 32, 4):\n"
            "        nibble = (word >> shift) & 0xF\n"
            "        out.append(nibble * scale + bias)\n"
            "    assert "
        ),
        "tests": [
            "assert solve(0x76543210, 1.0, 0.0) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]",
            "assert solve(0x11111111, 2.0, -1.0) == [1.0] * 8",
            "assert solve(0x0000000F, 0.5, 1.0)[0] == 8.5",
            "assert solve(0xF0000000, 1.0, 0.0)[-1] == 15.0",
        ],
    },
    {
        "id": "AlgoGen/13",
        "domain": "Algo-OpenEnded",
        "entry_point": "solve",
        "prompt": (
            "def solve(n, edges):\n"
            "    parent = list(range(n))\n"
            "    comps = n\n"
            "    def find(x):\n"
            "        _s = 0\n"
            "        while (parent[x] != x) and _s < 32:\n"
            "            _s += 1\n"
            "            parent[x] = parent[parent[x]]\n"
            "            x = parent[x]\n"
            "        if _s >= 32:\n"
            '            raise RuntimeError("UF cycle")\n'
            "        return x\n"
            "    for a, b in edges:\n"
            "        ra, rb = find(a), find(b)\n"
            "        if ra != rb:\n"
            "            parent[ra] = rb\n"
            "            comps = "
        ),
        "tests": [
            "assert solve(5, [(0,1),(1,2),(3,4)]) == 2",
            "assert solve(4, [(0,1),(1,2),(2,3),(0,3)]) == 1",
            "assert solve(3, []) == 3",
            "assert solve(4, [(0,1),(2,3)]) == 2",
        ],
    },
    {
        "id": "AlgoGen/29",
        "domain": "Algo-OpenEnded",
        "entry_point": "solve",
        "prompt": (
            "def solve(r1, m1, r2, m2):\n"
            "    for x in range(m1 * m2):\n"
            "        if (x % m1 == r1) and (x % m2 == r2):\n"
            "            assert "
        ),
        "tests": [
            "assert solve(2, 3, 3, 5) == 8",
            "assert solve(1, 2, 2, 3) == 5",
            "assert solve(0, 3, 0, 5) == 0",
            "assert solve(3, 4, 4, 5) == 19",
        ],
    },
    {
        "id": "AlgoGen/18",
        "domain": "Algo-OpenEnded",
        "entry_point": "kmp_lps",
        "prompt": (
            "def kmp_lps(pat: str) -> list[int]:\n"
            '    """Compute Knuth-Morris-Pratt longest proper prefix-suffix (LPS) array for pat."""\n'
            "    lps = [0] * len(pat)\n"
            "    length = 0\n"
            "    i = 1\n"
            "    while i < len(pat):\n"
            "        if pat[i] == pat[length]:\n"
            "            length += 1\n"
            "            lps[i] = length\n"
            "            i += 1\n"
            "        elif length != 0:\n"
            "            length ="
        ),
        "tests": [
            'assert kmp_lps("ABABCABAB") == [0, 0, 1, 2, 0, 1, 2, 3, 4]',
            'assert kmp_lps("AAAA") == [0, 1, 2, 3]',
            'assert kmp_lps("ABCDE") == [0, 0, 0, 0, 0]',
            'assert kmp_lps("AABAACAABAA") == [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]',
        ],
    },
    {
        "id": "AlgoGen/2",
        "domain": "Algo-OpenEnded",
        "entry_point": "fenwick_prefix_sum",
        "prompt": (
            "def fenwick_prefix_sum(nums: list[int], k: int) -> int:\n"
            '    """Build a 1-indexed Fenwick tree (Binary Indexed Tree) and query prefix sum of first k elements."""\n'
            "    n = len(nums)\n"
            "    tree = [0] * (n + 1)\n"
            "    for idx, val in enumerate(nums, start=1):\n"
            "        i = idx\n"
            "        while i <= n:\n"
            "            tree[i] += val\n"
            "            i += i & -i\n"
            "    s = 0\n"
            "    while k > 0:\n"
        ),
        "tests": [
            "assert fenwick_prefix_sum([1, 2, 3, 4, 5], 3) == 6",
            "assert fenwick_prefix_sum([1, 2, 3, 4, 5], 5) == 15",
            "assert fenwick_prefix_sum([10, -2, 7, 1], 2) == 8",
            "assert fenwick_prefix_sum([5, 5, 5], 0) == 0",
        ],
    },
    {
        "id": "AlgoGen/5",
        "domain": "Algo-OpenEnded",
        "entry_point": "max_subarray_kadane",
        "prompt": (
            "def max_subarray_kadane(nums: list[int]) -> int:\n"
            '    """Return the maximum contiguous subarray sum of non-empty list nums using Kadane algorithm."""\n'
        ),
        "tests": [
            "assert max_subarray_kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6",
            "assert max_subarray_kadane([1]) == 1",
            "assert max_subarray_kadane([5, 4, -1, 7, 8]) == 23",
            "assert max_subarray_kadane([-5, -2, -9]) == -2",
        ],
    },
    {
        "id": "AlgoGen/27",
        "domain": "Algo-OpenEnded",
        "entry_point": "kruskal_mst_weight",
        "prompt": (
            "def kruskal_mst_weight(n: int, edges: list[tuple[int, int, int]]) -> int:\n"
            '    """Given n vertices (0..n-1) and undirected weighted edges (u, v, w), return total weight of the Minimum Spanning Tree."""\n'
            "    parent = list(range(n))\n"
            "    def find(x: int) -> int:\n"
            "        while parent[x] != x:\n"
            "            parent[x] = parent[parent[x]]\n"
            "            x = parent[x]\n"
            "        return x\n"
            "    total = 0\n"
            "    for u, v, w in sorted(edges, key=lambda e: e[2]):\n"
        ),
        "tests": [
            "assert kruskal_mst_weight(4, [(0,1,10), (0,2,6), (0,3,5), (1,3,15), (2,3,4)]) == 19",
            "assert kruskal_mst_weight(3, [(0,1,1), (1,2,2), (0,2,3)]) == 3",
            "assert kruskal_mst_weight(2, [(0,1,7)]) == 7",
            "assert kruskal_mst_weight(4, [(0,1,1), (1,2,1), (2,3,1), (0,3,10)]) == 3",
        ],
    },
    {
        "id": "AlgoGen/30",
        "domain": "Algo-OpenEnded",
        "entry_point": "dijkstra_shortest_path",
        "prompt": (
            "import heapq\n\n"
            "def dijkstra_shortest_path(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int:\n"
            '    """Return shortest path distance from src to dst in a directed graph with non-negative weights (u, v, w), or -1 if unreachable."""\n'
            "    adj = [[] for _ in range(n)]\n"
            "    for u, v, w in edges:\n"
            "        adj[u].append((v, w))\n"
            "    dist = [float('inf')] * n\n"
            "    dist[src] = 0\n"
            "    pq = [(0, src)]\n"
            "    while pq:\n"
            "        d, u = heapq.heappop(pq)\n"
            "        if u == dst:\n"
            "            return d\n"
            "        if d > dist[u]:\n"
            "            continue\n"
            "        for v, w in adj[u]:\n"
        ),
        "tests": [
            "assert dijkstra_shortest_path(4, [(0,1,1), (1,2,2), (0,2,4), (2,3,1)], 0, 3) == 4",
            "assert dijkstra_shortest_path(3, [(0,1,5)], 0, 2) == -1",
            "assert dijkstra_shortest_path(3, [(0,1,2), (1,2,3)], 0, 0) == 0",
            "assert dijkstra_shortest_path(4, [(0,1,10), (0,2,3), (2,1,1), (1,3,2)], 0, 3) == 6",
        ],
    },
]


def extract_function_body(completion: str) -> str:
    """Truncate generated completion at the end of the target function body."""
    lines = completion.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if kept and stripped and not line.startswith((" ", "\t")):
            break
        kept.append(line)
    return "\n".join(kept) + "\n"


def evaluate_generated_batch_in_venv(
    problems: List[dict], raw_completions: List[str]
) -> List[Tuple[bool, int, int, str]]:
    """Compile and execute all generated functions against their unit tests inside `.venv/bin/python` (with pandas + numpy)."""
    payload = []
    for prob, raw_comp in zip(problems, raw_completions):
        body = extract_function_body(raw_comp)
        payload.append(
            {
                "id": prob["id"],
                "entry_point": prob["entry_point"],
                "full_code": prob["prompt"] + body,
                "tests": prob["tests"],
            }
        )
    runner_code = (
        "import heapq, json, math, signal, sys\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "def _timeout_handler(signum, frame):\n"
        "    raise TimeoutError('Execution timed out')\n"
        "signal.signal(signal.SIGALRM, _timeout_handler)\n"
        "items = json.loads(sys.stdin.read())\n"
        "out = []\n"
        "for item in items:\n"
        "    ns = {'np': np, 'pd': pd, 'math': math, 'heapq': heapq}\n"
        "    tests = item['tests']\n"
        "    try:\n"
        "        signal.alarm(2)\n"
        "        compiled = compile(item['full_code'], f\"<{item['id']}>\", 'exec')\n"
        "        exec(compiled, ns)\n"
        "        signal.alarm(0)\n"
        "    except Exception as e:\n"
        "        signal.alarm(0)\n"
        "        out.append([False, 0, len(tests), f'CompileError: {e}'])\n"
        "        continue\n"
        "    ep = item['entry_point']\n"
        "    if ep not in ns or not callable(ns[ep]):\n"
        "        out.append([False, 0, len(tests), 'MissingEntryPoint'])\n"
        "        continue\n"
        "    passed = 0\n"
        "    last_err = ''\n"
        "    for t_code in tests:\n"
        "        try:\n"
        "            signal.alarm(2)\n"
        "            exec(t_code, ns)\n"
        "            signal.alarm(0)\n"
        "            passed += 1\n"
        "        except Exception as e:\n"
        "            signal.alarm(0)\n"
        "            last_err = repr(e)\n"
        "    out.append([passed == len(tests), passed, len(tests), last_err])\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [VENV_PYTHON, "-c", runner_code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    raw_res = json.loads(proc.stdout.strip().splitlines()[-1])
    return [(bool(r[0]), int(r[1]), int(r[2]), str(r[3])) for r in raw_res]


def run_suite_generation(
    eng: MLXQwen38Engine,
    label: str,
    lora: dict | None = None,
    lora_scale: float = 2.0,
    max_new_tokens: int = 80,
    batch_size: int = 8,
) -> dict:
    """Run batched open-ended autoregressive generation across all 32 external benchmark problems."""
    t0 = time.time()
    all_completions: List[str] = []
    for i in range(0, len(EXTERNAL_BENCHMARK_PROBLEMS), batch_size):
        batch_probs = EXTERNAL_BENCHMARK_PROBLEMS[i : i + batch_size]
        prompts = [p["prompt"] for p in batch_probs]
        b_t0 = time.time()
        outs = eng.generate_batch(prompts, max_new_tokens=max_new_tokens, lora=lora, lora_scale=lora_scale)
        all_completions.extend(outs)
        print(
            f"  [{label}] Generated batch {i//batch_size + 1}/{len(EXTERNAL_BENCHMARK_PROBLEMS)//batch_size} ({len(prompts)} functions, max {max_new_tokens} tokens) in {time.time()-b_t0:.2f}s",
            flush=True,
        )

    domain_stats: Dict[str, dict] = {}
    total_fn_pass = 0
    total_assert_pass = 0
    total_assert_cnt = 0
    per_problem = []

    eval_results = evaluate_generated_batch_in_venv(EXTERNAL_BENCHMARK_PROBLEMS, all_completions)
    for prob, raw_comp, (ok, p_cnt, tot_cnt, err) in zip(
        EXTERNAL_BENCHMARK_PROBLEMS, all_completions, eval_results
    ):
        dom = prob["domain"]
        if dom not in domain_stats:
            domain_stats[dom] = {"fn_pass": 0, "fn_total": 0, "assert_pass": 0, "assert_total": 0}
        domain_stats[dom]["fn_total"] += 1
        domain_stats[dom]["fn_pass"] += int(ok)
        domain_stats[dom]["assert_pass"] += p_cnt
        domain_stats[dom]["assert_total"] += tot_cnt

        total_fn_pass += int(ok)
        total_assert_pass += p_cnt
        total_assert_cnt += tot_cnt
        per_problem.append(
            {
                "id": prob["id"],
                "domain": dom,
                "entry_point": prob["entry_point"],
                "passed": ok,
                "assertions_passed": p_cnt,
                "assertions_total": tot_cnt,
                "error": err,
                "generated_body": extract_function_body(raw_comp),
            }
        )

    n_probs = len(EXTERNAL_BENCHMARK_PROBLEMS)
    summary = {
        "label": label,
        "pass_at_1": round(100.0 * total_fn_pass / n_probs, 2),
        "fn_pass": total_fn_pass,
        "fn_total": n_probs,
        "assert_acc": round(100.0 * total_assert_pass / total_assert_cnt, 2),
        "assert_pass": total_assert_pass,
        "assert_total": total_assert_cnt,
        "elapsed_sec": round(time.time() - t0, 2),
        "domains": domain_stats,
        "per_problem": per_problem,
    }
    print(
        f"==> [{label}] Overall Open-Ended Pass@1: {total_fn_pass}/{n_probs} ({summary['pass_at_1']:.2f}%) | "
        f"Assertions: {total_assert_pass}/{total_assert_cnt} ({summary['assert_acc']:.2f}%) in {summary['elapsed_sec']:.1f}s"
    )
    for dom, st in domain_stats.items():
        print(
            f"    - {dom:18s}: Pass@1 = {st['fn_pass']}/{st['fn_total']} ({100.0*st['fn_pass']/st['fn_total']:.2f}%) | "
            f"Assertions = {st['assert_pass']}/{st['assert_total']} ({100.0*st['assert_pass']/st['assert_total']:.2f}%)"
        )
    return summary


def main():
    print("=" * 92)
    print("EXTERNAL OPEN-ENDED AUTOREGRESSIVE BENCHMARK: HumanEval + MBPP + DS-1000 + Algo (40 Functions)")
    print("=" * 92)
    eng = MLXQwen38Engine()
    ckpt_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        "Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
    )
    with open(os.path.join(ckpt_dir, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    lora_scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    lora_weights = mx.load(os.path.join(ckpt_dir, "adapter_model.safetensors"))

    base_res = run_suite_generation(eng, "Unmodified Base Qwen/Qwen3.8-27B", lora=None, max_new_tokens=104)
    lora_res = run_suite_generation(
        eng,
        "Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
        lora=lora_weights,
        lora_scale=lora_scale,
        max_new_tokens=104,
    )

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "external_humaneval_ds1000_report.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base": base_res, "realcoder_v3": lora_res}, f, indent=2)
    print(f"\n[SAVED] External benchmark report written to {out_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()

