import math

try:
  with open('/answer.py') as f:
    exec(f.read())
  _rv = {{call}}
except Exception as e:
  _rv = { 'error': str(e), 'exc_type': type(e).__name__ }
import json
print("###",json.dumps(_rv))