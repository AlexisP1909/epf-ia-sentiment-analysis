import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet',
          'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet(
    "hf://datasets/google-research-datasets/poem_sentiment/" + splits["train"])
