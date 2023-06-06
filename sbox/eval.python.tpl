import math
import types
try:
  with open('/answer') as f:
    exec(f.read())
  _rv = {{call}}
except Exception as e:
  _rv = { 'error': str(e), 'exc_type': type(e).__name__ }
if isinstance(_rv, types.GeneratorType):
  _rv = list(_rv)
import json
print("###",json.dumps(_rv))