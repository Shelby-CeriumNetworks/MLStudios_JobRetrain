import os
import pandas as pd
from sklearn.datasets import load_iris

def load_dataset():
    csv_path = os.getenv("DATA_PATH","data/training.csv")
    if os.path.exists(csv_path):
        print(f"Found dataset at {csv_path}, loading...")
        data = pd.read_csv(csv_path)
        print(f"Loaded dataset from {csv_path}")
        ## Assume last column is the target (true for diabetes dataset)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values        
        return X,y, {"source": "csv", "path": csv_path}
    ## if not found, load from sklearn
    iris = load_iris()
    return iris.data, iris.target, {"source": "sklearn", "dataset": "iris"}

## test to see if it works
X,y,meta = load_dataset()
print("Dataset shape:", X.shape, y.shape)
print("Metadata:", meta)