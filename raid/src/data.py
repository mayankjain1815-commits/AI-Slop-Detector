import pandas as pd

def sample_data(n_sample: int, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv('./raid-dataset/train_none.csv')
    df = df[["id", "domain", "model", "generation"]].sample(n=n_sample, random_state=random_state)

    return df
