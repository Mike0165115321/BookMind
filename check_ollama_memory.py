import urllib.request
import json

try:
    req = urllib.request.Request("http://localhost:11434/api/ps")
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error checking Ollama ps: {e}")
