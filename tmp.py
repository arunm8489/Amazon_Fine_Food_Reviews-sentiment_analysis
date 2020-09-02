import json,requests

# local url
url = 'http://127.0.0.1:5000'

data = {'review': 'food has good taste'}
data = json.dumps(data)
response = requests.post(url,data)

print(response.json())