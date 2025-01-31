from azureml.core import Workspace
from azureml.mlflow import get_mlflow_tracking_uri
from dotenv import load_dotenv
import openai
import os
import mlflow
import requests

# Load environment variables from .env file
load_dotenv()

# Retrieve the secrets using os.getenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")
openai_api_version = os.getenv("OPENAI_API_VERSION")
subscription_id= os.getenv("SUBSCRIPTION_ID")
resource_group= os.getenv("WORKSPACE_NAME")
workspace_name= os.getenv("RESOURCE_GROUP")


# Set up OpenAI API configuration for Azure
openai.api_key = openai_api_key
openai.api_base = openai_api_base
openai.api_version = openai_api_version

# Construct the URL for the Azure OpenAI endpoint
url = f"{openai.api_base}/openai/deployments/{openai_deployment_name}/chat/completions?api-version={openai_api_version}"

# Set headers for authentication and content type
headers = {
    "Content-Type": "application/json",
    "api-key": openai_api_key
}
endpoint_name = "zahee-m49xcudr-eastus2"

# Get the workspace from Azure
ws = Workspace(subscription_id, resource_group, workspace_name)

# Set the tracking URI to the Azure ML workspace
mlflow.set_tracking_uri(get_mlflow_tracking_uri(ws))

# Function to mimic the `client.predict` behavior using requests
def predict(inputs):
    prompt = inputs.get("prompt")
    max_tokens = inputs.get("max_tokens", 100)

    # Prepare the payload for the Azure OpenAI service
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }

    # Make the request to the Azure OpenAI endpoint
    try:
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data['choices'][0]['message']['content']
        else:
            print(f"Error: {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Prepare inputs for prediction
inputs = {
    "prompt": "How is Pi calculated? Be very concise.",
    "max_tokens": 100,
}

# Call the prediction function
prediction = predict( inputs)

# Print the prediction result
print(f"Prediction: {prediction}")

# Call the prediction function
prediction = predict(inputs)

# Print the prediction result
print(f"Prediction: {prediction}")
# Optionally, log the prediction and other details to MLflow
# Log a simple experiment to test
mlflow.start_run()
mlflow.log_param("openai_endpoint", endpoint_name)
mlflow.log_param("openai_deployment_name", openai_deployment_name)
mlflow.log_param("prompt", inputs["prompt"])
mlflow.log_metric("response_length", len(prediction))
mlflow.end_run()

print("MLflow experiment logged to Azure")

# Optionally print additional info
print(f"Prediction: {prediction}")