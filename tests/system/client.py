#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json


def call_score_endpoint(question, chat_history=None):
    """
    Call the score endpoint of the API to get a response from the chat model.
    
    Args:
        question (str): The question to ask the model
        chat_history (list, optional): Chat history for context. Defaults to empty list.
    
    Returns:
        dict: The response from the API
    """
    if chat_history is None:
        chat_history = []

    url = "http://127.0.0.1:5000/score"

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    payload = {"question": question, "chat_history": chat_history}

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    # Example usage
    result = call_score_endpoint("Compute 1 + 1.")
    print(result)
