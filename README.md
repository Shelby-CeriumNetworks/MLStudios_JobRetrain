## ML Studio Retraining

## Purpose
Machine learning models don’t stay perfect forever. As new data is collected, customer behavior shifts, or environments change, models can start to drift — their predictions drift away from reality and accuracy declines.

This repository provides a simple, repeatable pattern to handle that problem:

This repository demonstrates **automated, repeatable retraining** with:
- **Data loading** (from a CSV if provided, otherwise a built-in dataset)
- **Scheduled job** (via GitHub Actions cron)
- **Model evaluation** (accuracy, auc)
- **Comparison with previous evaluations** and model promotion if improved


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

3. If there is a specific dataset to test- enter the pathway to it by using
    ```bash
    setx DATA_PATH "<pathway>"
    ```

4. Clone the repository 

5. Alter the model as fit- specifically change the required training job type in `train.py`

6. Go into cloned GitHub Repo and press Action tab

7. Click on Scheduled Retrain and run workflow

8. The workflow should complete and the icon should be green. At this step, you may now be able to see the artifacts folder and information.


## Pipeline Pathway

1. Load in dataset: `dataloader.py`. There is a few options to perform this

- upload the data pathway into a environment variable "DATA_PATH"
- upload data into a folder pathway "data/training.csv"
- if no data is provided, the repo will fall back onto the 'iris' dataset (multiclass)

2. Training: `train.py` 
- Trains a classification (can be configured depending on job) and saves a model artifact.
- If binary classification will record AUC, if non-binary classification it will skip this metric.

3. Compare models: `compare.py`

a. Compares the most recent model with the best model in the artifacts using evaluation criteria set in `train.py`.

b. If the current model is better it will override and will promote the new best model.

4. Testing run: `testingrun.py` 
- will start a new run connecting the above pieces to ensure the pipeline is complete and compatible

5. Pipeline Setup: `pipeline.py` combines all the elements and is what the GitHub action uses for automated runs

6. Scheduled Job: `.github/workflows/retrain.yml` 
- Scheduled every Sunday at 0 UTC to run a classification through ML Studios on the model and record/evaluates the model
- Utilizes `pipeline.py` in order to combine all the steps and create a successful run. 



