import urllib.request
import json
try:
    url = "https://huggingface.co/api/models/timm/vit_base_patch16_224.inat2021"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(len(data.get('config', {}).get('id2label', {})))
except Exception as e:
    print(e)
