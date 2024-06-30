import requests

url = 'http://localhost:1313'
response = requests.get(url)
print(response.status_code)
print(response.text)
