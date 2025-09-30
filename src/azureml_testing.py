import os
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient

def get_credential():
    try:
        cred = DefaultAzureCredential()
        cred.get_token("https://management.azure.com/.default")
        return cred
    except Exception:
        return InteractiveBrowserCredential()

credential = get_credential()

print(os.environ.get("AZURE_SUBSCRIPTION_ID"))

subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group  = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name  = os.environ["AZURE_WORKSPACE_NAME"]

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# print compute to make sure connection works
for compute in ml_client.compute.list():
    print(compute.name, compute.type)
