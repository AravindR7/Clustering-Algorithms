from typing import Any
from layer import Dataset


def build_feature(sdf: Dataset("customers")) -> Any:

    data = sdf.to_pandas()
    input_data = data[['SpendingScore', 'Age']]
    return input_data
