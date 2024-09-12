import math
import types
def stub(*args, **kwargs):
  return "0"
input = stub
try:
  with open('/answer') as f:
    exec(f.read())
  _rv = {{call}}
except Exception as e:
  _rv = { 'error': str(e), 'exc_type': type(e).__name__ }
if isinstance(_rv, types.GeneratorType):
  _rv = list(_rv)
if isinstance(_rv, list):
  _rv = [int.from_bytes(x, 'little') if isinstance(x, bytes) else x for x in _rv]
import json
print("###",json.dumps(_rv))