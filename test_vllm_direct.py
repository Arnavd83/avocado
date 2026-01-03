#!/usr/bin/env python3
"""
Direct test of vLLM server without LiteLLM to see raw responses.
"""

import requests
import json

# Test the vLLM server directly
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-local-test",
    "Content-Type": "application/json"
}

payload = {
    "model": "meta-llama/Meta-Llama-3.1-8B",
    "messages": [
        {
            "role": "user",
            "content": "Which is better?\n\nOption A: Apple\nOption B: Orange\n\nAnswer with only 'A' or 'B'."
        }
    ],
    "temperature": 0.0,
    "max_tokens": 10
}

print("Testing vLLM server directly...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}\n")

response = requests.post(url, headers=headers, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Response Headers: {dict(response.headers)}\n")

if response.status_code == 200:
    data = response.json()
    print(f"Full Response: {json.dumps(data, indent=2)}\n")
    
    if 'choices' in data and len(data['choices']) > 0:
        content = data['choices'][0]['message']['content']
        print(f"✓ Generated text: '{content}'")
        print(f"  Length: {len(content)} characters")
        print(f"  Repr: {repr(content)}")
    else:
        print("✗ No choices in response!")
else:
    print(f"✗ Error: {response.text}")

