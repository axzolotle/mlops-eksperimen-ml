import pandas as pd
from pathlib import Path
import kagglehub
import os

dataset_path = kagglehub.dataset_download("amanalisiddiqui/fraud-detection-dataset")

csv_path = os.path.join(dataset_path, "AIML Dataset.csv")

# PATH CONFIG
RAW_PATH = Path(csv_path)
OUTPUT_PATH = Path("preprocessing/data_clean.csv")

# LOAD DATA
def load_data(path: Path, nrows : int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)

# PREPROCESSING
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encoding
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Drop duplicates
    df = df.drop_duplicates()

    # Frequency encoding
    df["dest_freq"] = df["nameDest"].map(df["nameDest"].value_counts())
    df["orig_freq"] = df["nameOrig"].map(df["nameOrig"].value_counts())

    # Drop identifier columns
    df = df.drop(columns=["nameDest", "nameOrig"])

    # Drop nulls (aman)
    df = df.dropna()

    return df

# MAIN
def main():
    df = load_data(RAW_PATH, nrows=100000)

    df_clean = preprocess(df)

    x = 500000
    n = len(df_clean)
    df_sample = (
        df_clean.sample(n=min(x, n), random_state=42)
    )
    df_sample.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
