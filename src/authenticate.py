from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command


subscription_id = "cf9f3e65-7780-4a9d-bbdd-ed2351502118"
resource_group = "rg-datasciencefundamentals-sames"
workspace = "mlw-dsfundamentals-sames"

#authenticate
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    experiment_name="train-model"
)

# connect to workspace and submit job
returned_job = ml_client.create_or_update(job)