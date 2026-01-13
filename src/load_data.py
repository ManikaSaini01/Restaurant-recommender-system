from datasets import load_dataset
import pandas as pd

def load_hf_dataset(dataset_name: str):
    """
    Load the dataset from Hugging Face
    and convert to a Pandas DataFrame.
    """
    ds = load_dataset(dataset_name)
    df = ds["train"].to_pandas()
    return df