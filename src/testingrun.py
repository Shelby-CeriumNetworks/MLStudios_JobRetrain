import uuid
from src.train import notebook_train_model
from src.compare import compare_and_promote

# Train the model
model, X_test, y_test, meta, metrics = notebook_train_model()

# Create a run record
run_info = {
    "run_id": str(uuid.uuid4()),
    "metrics": metrics, 
    "meta": meta
}

# Compare against history and possibly promote
result = compare_and_promote(run_info)

print("Comparison result:", result)
