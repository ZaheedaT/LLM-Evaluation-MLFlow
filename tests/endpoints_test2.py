#import unittest


#class MyTestCase(unittest.TestCase):
#	def test_something(self):
#		self.assertEqual(True, False)  # add assertion here


#if __name__ == '__main__':
#	unittest.main()

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

# Set up OpenAI API configuration
openai.api_key = openai_api_key
openai.api_base = openai_api_base
openai.api_version = openai_api_version

print("------------ENVS------------", openai_deployment_name, openai_api_base,openai_api_version, openai_api_key)
url = f"{openai.api_base}/openai/deployments/{openai_deployment_name}/chat/completions?api-version={openai_api_version}"
print("------------THE URL WE ARE USING ----------------", url)

# Set headers for authentication and content type
headers = {
    "Content-Type": "application/json",
    "api-key": openai.api_key
}

# Set up MLflow if you need tracking
# # Use if you have a remote MLflow server

# Example: Defining the endpoint name you want to test
endpoint_name = "zahee-m49xcudr-eastus2"  # replace this!

# Prepare the payload for the chat model
payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 50
}


# Make the API request
response = requests.post(url, headers=headers, json=payload)

# Check the response status and print details
if response.status_code == 200:
    data = response.json()
    print("Response from OpenAI:", data['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)