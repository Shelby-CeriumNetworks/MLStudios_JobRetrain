import uuid, json
from src.train import notebook_train_model
from src.compare import compare_and_promote

def main():
    ## train the model
    model, X_test, y_test, meta, metrics = notebook_train_model()
    run_info = {
        "run_id": str(uuid.uuid4()),
        "metrics": metrics,
        "meta": meta 
    }
    result = compare_and_promote(run_info)

    print(json.dumps({
        "run_id": run_info["run_id"],
        "metrics": metrics,
        "compare_result": result
    }, indent=2))

if __name__ == "__main__":
    main()
