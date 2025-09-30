## ML Studio Retraining

This repository demonstrates **automated, repeatable retraining** with:
- **Data loading** (from a CSV if provided, otherwise a built-in dataset)
- **Scheduled job** (via GitHub Actions cron)
- **Model evaluation** (accuracy, auc)
- **Comparison with previous evaluations** and model promotion if improved

The main purpose of this repo is to showcase proof that a model can be retrained regularly with minimal effort to address model sway some clients may experience.

## How to use
1. Create a python environment and install dependencies:
    ```bash
    py -m pip install -r requirements.txt
    ```

2. Connect to ML Studio using Azure Credentials and environment variabes

a. first need to enter environment variables
    ```bash
    setx AZURE_SUBSCRIPTION_ID "<sub_id>"
    setx AZURE_RESOURCE_GROUP "<resourge_group_name>"
    setx AZURE_WORKSPACE_NAME "<workspace_name>"
    ```

b. check variables have been updated. In a new power shell terminal 
    ```bash
    echo $env:AZURE_SUBSCRIPTION_ID
    echo $env:AZURE_RESOURCE_GROUP
    echo $env:AZURE_WORKSPACE_NAME
    ```

c. Run azureml_testing.py to test connection. Should print out available computes in ML Studio.

3. Load in dataset: `dataloader.py`. There is a few options to perform this

- upload the data pathway into a environment variable "DATA_PATH"
- upload data into a folder pathway "data/training.csv"

4. Training: `train.py` 
- Trains a classification (can be configured depending on job) and saves a model artifact.

5. Compare models: `compare.py`

a. Compares the most recent model with the best model in the artifacts using evaluation criteria set in `train.py`.

b. If the current model is better it will override and will promote the new best model.



