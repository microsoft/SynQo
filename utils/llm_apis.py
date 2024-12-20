import requests
import urllib.request
import json
import os
import ssl
import time
import re
import json

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import TextCategory
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions


def getSaftyCheck(content):
    base_url = "" # to be added
    api_key = "" # to be added
    client = ContentSafetyClient(base_url, AzureKeyCredential(api_key))

    # Construct a request
    request = AnalyzeTextOptions(text=content)

    # Analyze text
    try:
        response = client.analyze_text(request)
    except HttpResponseError as e:
        print("Analyze text failed.")
        if e.error:
            print(f"Error code: {e.error.code}")
            print(f"Error message: {e.error.message}")
        
        return False

    for item in response.categories_analysis:
        if item.severity > 0:
            return False

    return True
    

def getCompletionGPT4(
    messages, model_name="gpt-4-turbo", data_params=None, retry=True):  
    base_url = "" # to be added
    version = "" # to be added
    model = "" # to be added
    url = (
        f"{base_url}/openai/deployments/{model}/chat/completions?api-version={version}"
    )
    api_key ="" # to be added
    headers = {"Content-Type": "application/json", "api-key": api_key}
    data = {
        "messages": json.loads(
            json.dumps(messages)
        ),  # bad characters in prompt like ' or could cause errors. converting to json and reconverting to object was easiest but not clean solution that i chose temprorary.

    }
    if data_params is not None:
        data = data | data_params  # merging params with data.
        n = data["n"]
    else:
        n = 1
    nextResponse = requests.post(url, headers=headers, json=data)
    if n == 1:
        if "tool_calls" in nextResponse.json()["choices"][0]["message"]:
            res = nextResponse.json()["choices"][0]["message"]["tool_calls"]
            return res if getSaftyCheck(res) else ""

        res = nextResponse.json()["choices"][0]["message"]["content"]
        return res if getSaftyCheck(res) else ""
    else:
        return [c["message"]["content"] for c in nextResponse.json()["choices"] if getSaftyCheck(c["message"]["content"])]        
