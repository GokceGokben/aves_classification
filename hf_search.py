import urllib.request
import json

url = "https://huggingface.co/api/models?search=bird"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    
models = sorted(data, key=lambda x: x.get('downloads', 0), reverse=True)[:10]
for m in models:
    print(f"Model: {m['id']} - Downloads: {m.get('downloads', 0)}")
