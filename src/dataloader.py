import os
import pandas as pd
from sklearn.datasets import load_iris

def load_dataset():
    csv_path = os.getenv("DATA_PATH","data/training.csv")
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        print(f"Loaded dataset from {csv_path}")
        ## Assume last column is the target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X,y, {"source": "csv", "path": csv_path}
    ## if not found, load from sklearn
    iris = load_iris()
    return iris.data, iris.target, {"source": "sklearn", "dataset": "iris"}