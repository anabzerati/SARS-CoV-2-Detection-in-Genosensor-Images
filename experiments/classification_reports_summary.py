import pandas as pd
import numpy as np
import glob
import os

def parse_report(path):
    """
    Parse a classification report text file and compute mean precision, recall, and F1-score across classes.

    Args:
        path (str):
            Path to the classification report text file.

    Returns:
        pd.DataFrame:
            A single-row DataFrame containing the mean precision, recall, and F1-score.
    """
    prec_list, rec_list, f1_list = [], [], []
    
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:  # skip empty lines
                continue
            if parts[0] not in ['precision', 'accuracy', 'macro', 'weighted']:
                try:
                    # Expected format: [class, precision, recall, f1-score, support]
                    _, prec, rec, f1, _ = parts
                    prec_list.append(float(prec))
                    rec_list.append(float(rec))
                    f1_list.append(float(f1))
                except ValueError:
                    # skip lines that do not match expected format
                    continue
    
    # compute averages
    mean_prec = np.mean(prec_list)
    mean_rec = np.mean(rec_list)
    mean_f1 = np.mean(f1_list)

    # create DataFrame
    df = pd.DataFrame([{
        "precision": mean_prec,
        "recall": mean_rec,
        "f1-score": mean_f1
    }])

    return df


# Read all report text files
files = glob.glob("results/*_classification_report.txt")

dfs = []
for file in files:
    name = os.path.basename(file).replace("_classification_report.txt", "")
    
    # handle different naming patterns for convnext models
    if name.startswith("convnext"):
        model1, model2, dataaug, _ = name.split("_", 3)
        model = model1 + model2
    else:
        model, dataaug, _ = name.split("_", 2)

    print(model, dataaug)
    
    df = parse_report(file)
    df["model"] = model
    df["augmentation"] = dataaug

    print(df)

    dfs.append(df)

# concatenate all DataFrames
df_all = pd.concat(dfs, ignore_index=True)

# aggregate statistics by model and augmentation
df_stats = (
    df_all.groupby(["model", "augmentation"])
    [["precision", "recall", "f1-score"]]
    .agg(["mean", "std"])
)

print(df_stats)
