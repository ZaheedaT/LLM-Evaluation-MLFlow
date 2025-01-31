# Rewriting the PDF generation code without special characters to avoid encoding issues

from fpdf import FPDF

# Create a PDF document
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Set title
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, "Deploying an MLflow Model to Production on Azure", ln=True, align='C')
pdf.ln(10)

# Add introduction text
pdf.set_font("Arial", size=12)

from fpdf import FPDF
import unicodedata

# Function to clean text by removing unsupported characters
def remove_invalid_chars(text):
    # Normalize to NFKD to decompose characters into simpler forms
    text = unicodedata.normalize('NFKD', text)
    # Encode to latin-1, replacing unsupported characters with a placeholder
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text



# Example text with special characters
text_with_unicode = "Here's a text with special characters like ‘ and ’."

# Clean the text to remove unsupported characters
safe_text = remove_invalid_chars(text_with_unicode)


# Add the cleaned text to the PDF
pdf.multi_cell(0, 10, safe_text)

# Output the PDF
pdf_output_path = "output.pdf"
pdf.output(pdf_output_path)

print(f"PDF generated: {pdf_output_path}")


text = """When your model is ready for production and you want to deploy it using Azure Machine Learning (AML) and MLflow,
the steps shift from experiment tracking to managing and deploying models. Below is a detailed process for deploying a model in
a production environment using Azure Machine Learning (AML) and MLflow."""
safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)
pdf.ln(10)

# Step 1: Register the Model in Azure ML
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "1. Register the Model in Azure ML", ln=True)
pdf.set_font("Arial", size=12)
text = """Once your model is trained, you'll need to register the model in Azure Machine Learning.
This allows you to version and manage models for production."""

safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)

# Example of Registering the Model
example_code = """
import mlflow
from azureml.core import Workspace
from azureml.mlflow import get_mlflow_tracking_uri
from azureml.core.model import Model

# Set up the Azure ML workspace
ws = Workspace(subscription_id="your_subscription_id", resource_group="your_resource_group", workspace_name="your_workspace_name")

# Get the tracking URI for Azure ML
mlflow.set_tracking_uri(get_mlflow_tracking_uri(ws))

# Register the model to Azure ML from the MLflow run
model_uri = "runs:/<run_id>/model"  # Path to your model
model_name = "my_deployed_model"

# Register the model with Azure ML
model = Model.register(workspace=ws,
                       model_name=model_name,
                       model_path=model_uri)

print(f"Model registered: {model.name}, version: {model.version}")
"""

pdf.set_font("Courier", size=10)
safe_text = remove_invalid_chars(example_code)

pdf.multi_cell(0, 10, example_code)
pdf.ln(10)

# Step 2: Prepare the Model for Deployment
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "2. Prepare the Model for Deployment", ln=True)
pdf.set_font("Arial", size=12)
text = """Now that your model is registered, you need to prepare it for deployment.
For production, models can be deployed as web services on Azure Kubernetes Service (AKS), Azure Container Instances (ACI),
or other supported compute environments."""

safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)

# Scoring Script Example
scoring_script = """
import json
import mlflow
import azureml
from azureml.core.model import Model
import numpy as np
import pandas as pd

def init():
    global model
    model_path = Model.get_model_path('my_deployed_model')  # Retrieve model from Azure ML
    model = mlflow.pyfunc.load_model(model_path)  # Load the MLflow model

def run(data):
    try:
        # Convert input JSON to a DataFrame
        data = pd.DataFrame(json.loads(data)['data'])

        # Predict using the MLflow model
        predictions = model.predict(data)

        # Return predictions as JSON
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
"""

pdf.set_font("Courier", size=10)
safe_text = remove_invalid_chars(scoring_script)

pdf.multi_cell(0, 10, safe_text)
pdf.ln(10)

# Step 3: Deploy the Model as a Web Service
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "3. Deploy the Model as a Web Service", ln=True)
pdf.set_font("Arial", size=12)
text = """Once you have the scoring script and environment configuration ready, you can deploy your model as a web service.
You can choose between Azure Container Instances (ACI) for light, on-demand deployments or Azure Kubernetes Service (AKS)
for more scalable, production-grade deployments."""
safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)


# Example of Deployment with ACI
aci_code = """
from azureml.core import Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model
from azureml.core import ScriptRunConfig

# Define the environment
env = Environment.from_conda_specification(name="my_model_environment", file_path="env.yml")

# Configure the ACI deployment
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model to ACI
service = Model.deploy(workspace=ws,
                       name="my-production-model",
                       models=[model],
                       inference_config=env,
                       deployment_config=aci_config)

service.wait_for_deployment(True)
print(f"Service state: {service.state}")
"""

pdf.set_font("Courier", size=10)
safe_text = remove_invalid_chars(aci_code)
pdf.multi_cell(0, 10, safe_text)

pdf.ln(10)

# Step 4: Monitor and Manage the Model in Production
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "4. Monitor and Manage the Model in Production", ln=True)
pdf.set_font("Arial", size=12)
text = """Once your model is deployed, you will want to monitor its performance, manage its versions,
and potentially retrain it as new data comes in."""

safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)

# Step 5: Updating the Model in Production
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "5. Updating the Model in Production", ln=True)
pdf.set_font("Arial", size=12)
text = """If you need to update the model in production, follow these steps:
1. Register a New Version: When you retrain or fine-tune the model, register it again as a new version.
2. Deploy the New Version: Use Azure ML’s deployment methods to push the new model version to production.
3. Traffic Management: Azure allows you to manage the traffic between different versions of the model."""

safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)
pdf.ln(10)

# Step 6: Secure the Model Endpoint
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "6. Secure the Model Endpoint", ln=True)
pdf.set_font("Arial", size=12)
text = """To secure the deployed web service, you can use authentication mechanisms like Azure Active Directory (AAD),
which ensures that only authorized users can access the API endpoint."""

safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)

# Example of Securing the API Endpoint
security_code = """
# Set up authentication using Azure AD tokens for the web service
from azureml.core.webservice import Webservice

service = Webservice(workspace=ws, name="my-production-model")
service.update(enable_app_insights=True)  # Enable monitoring with App Insights

# Use the web service with authentication (using the token)
token = service.get_keys()[0]
headers = {'Authorization': 'Bearer ' + token}

# Now you can use this token to securely call the endpoint
response = requests.post(service.scoring_uri, json=data, headers=headers)
"""

pdf.set_font("Courier", size=10)
safe_text = remove_invalid_chars(security_code)
pdf.multi_cell(0, 10, safe_text)
pdf.ln(10)

# Step 7: Access the Deployed Model via API
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "7. Access the Deployed Model via API", ln=True)
pdf.set_font("Arial", size=12)
text = """Once the model is deployed, you can interact with it through an API endpoint.
For example, to make a prediction, you can send a POST request to the endpoint."""

safe_text = remove_invalid_chars(text)
pdf.multi_cell(0, 10, safe_text)

# Example Code to Call the API
api_call_code = """
import requests
import json

# Data to send to the model
data = {"data": [[5.1, 3.5, 1.4, 0.2]]}  # Example for Iris dataset

# Send the request to the deployed endpoint
response = requests.post(service.scoring_uri, json=data)
print("Prediction:", response.json())
"""

pdf.set_font("Courier", size=10)
safe_text = remove_invalid_chars(api_call_code)
pdf.multi_cell(0, 10, safe_text)
pdf.ln(10)

# Save the PDF
pdf_output_path = "/mnt/c/Users/zahee/Downloads/production_model_deployment_on_azure.pdf"
pdf.output(pdf_output_path)

print(pdf_output_path)
