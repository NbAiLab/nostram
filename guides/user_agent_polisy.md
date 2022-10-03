# User Agent Policy
All data collection in the project should be done as openly as absolutely possible. All requests should respect robots.txt-policies, and also identify the agent according to recommended standards. Below is the standard http-header for all requests. Feel free to use your own email address if you want to be contacted directly:

```python
import requests

headers = {
    'User-Agent': 'National Library of Norway - AiLab - NoSTraM Procect - User Agent v 1.0',
    'From': 'per.kummervold@nb.no' 
    }

response = requests.get(url, headers=headers)
```

If the script can cause unnessasary load on other systems, it is recommended to add a small delay between each request:

```python
import time

time.sleep(0.1) 
````

