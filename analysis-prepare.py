import json
import sys
import os
from jinja2 import Template

task_prompt = "Write a {{language}} function {{Signature}} {{Input}} that returns {{Output}}"

TEST_LANGUAGE = sys.argv[1]
files = sys.argv[2:]
id = 0
out = {}
models = []

for file in files:
    tags = os.path.basename(file).replace('.ndjson', '').split('_')
    prompt = tags[3]
    params = tags[5]
    model = tags[6]
    models.append({'prompt': prompt, 'params': params, 'model': model, 'id': id})
    #print(models[-1])
    results = [json.loads(line) for line in open(file)]
  
    for r in results:
        if r['language'] != TEST_LANGUAGE:
            continue

        testid = r['name']+'-'+r['language']
        task = Template(task_prompt).render(**r)
        if testid not in out:
            out[testid] = { 'results': {}, 'task': task, 'language': r['language'] }

        check_summary = f"{r['status']} correct {r['passed']}/{r['total']}"
        passing_tests = ''
        failing_tests = ''
        for c in r['checks']:
            if c['status'] == 1:
                passing_tests += f"PASS {c['assert']} == {c['eq']}\n"
            else:
                failing_tests += f"FAIL {c['assert']} != {c['eq']} got {c['got']}\n"

        out[testid]['results'][id] = {
            'check_summary': check_summary,
            'passing_tests': passing_tests,
            'failing_tests': failing_tests,
            'code': r['code']
        }

    id = id + 1

print(json.dumps({ 'tests': out, 'models': models }, indent=2))
