import requests
import json

url="http://127.0.0.1:5000/predict"

with open("example.json") as f:
    data=json.load(f)

response=requests.post(url,json=data)
print(response.json())