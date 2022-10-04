# User Agent Policy
All data collection in the project should be done as openly as absolutely possible. All requests should respect robots.txt-policies, and also identify the agent according to recommended standards. Below is the standard http-header for all requests:

```python
import requests

headers = {
    'User-Agent': 'National Library of Norway - AiLab - NoSTraM Project - User Agent v1.0',
    'From': 'ai-lab@nb.no' 
    }

# Then use one of the methods below:
response = requests.get(url, headers=headers)

# Or
with requests.Session() as session:
    session.headers.update(HEADERS)
    session.get(...)


```

If the script can cause unnessasary load on other systems, it is recommended to add a small delay between each request:

```python
import time

time.sleep(0.1) 
````

