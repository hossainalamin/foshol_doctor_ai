import pandas as pd

FILE_PATH = "dataset/bangla_rice_disease_dataset_1800_with_split.csv"

df = pd.read_csv(FILE_PATH)

print("Total rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nSample rows:")
print(df[["text", "label", "split"]].head(10))