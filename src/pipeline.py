import uuid, json
from src.train import notebook_train_model
from src.compare import compare_and_promote
import numpy as np

def to_jsonable(x):
    if isinstance(x, np.generic):         
        return x.item()
    if isinstance(x, np.ndarray):         
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x

def main():
    ## train the model
    model, X_test, y_test, meta, metrics = notebook_train_model()
    run_info = {
        "run_id": str(uuid.uuid4()),
        "metrics": metrics,
        "meta": meta 
    }
    result = compare_and_promote(run_info)

    print(json.dumps(to_jsonable({
        "run_id": run_info["run_id"],
        "metrics": metrics,
        "compare_result": result
    }), indent=2))


if __name__ == "__main__":
    main()
