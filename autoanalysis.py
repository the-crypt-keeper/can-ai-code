import json
import glob
import sys
import os
from jinja2 import Template
from langchain import LLMChain, PromptTemplate

TEST_LANGUAGE = "javascript"

header_prompt = """
You are going to evaluate the results of language models on a {{language}} programming challenge.
The challenge given to each model is to write a {{language}} function {{Signature}} {{Input}} that returns {{Output}}.
You will be provided the code produced by each model.
Automated tests have evaluated the performance of each model on the challenge, a list of passing and failing tests will also be provided.
Compare and constract the solutions each model produced, highlighting any differences in test results and provide a final summary of the results.
"""

model_prompt = """---
Model: {{id}}
Test Result: {{status}} correct {{passed}}/{{total}}
Test Details:
{{passing_tests}}{{failing_tests}}
Code:
```{{language}}
{{code}}
```
"""

footer_prompt = """---

Analysis:"""

files = glob.glob('results/eval*precise*orca-mini*.ndjson' if len(sys.argv) == 1 else sys.argv[1])
files = [ files[1], files[2], files[0] ]
id = 0
out = {}
models = []

for file in files:
    tags = os.path.basename(file).replace('.ndjson', '').split('_')
    prompt = tags[3]
    params = tags[5]
    model = tags[6]
    models.append({'prompt': prompt, 'params': params, 'model': model, 'id': id})
    print(models[-1])
    results = [json.loads(line) for line in open(file)]
  
    for r in results:
        if r['language'] != TEST_LANGUAGE:
            continue

        testid = r['name']+'-'+r['language']
        if testid not in out:
            out[testid] = { 'prompt': Template(header_prompt).render(**r), 'models': {} }

        check_summary = f"{r['status']} correct {r['passed']}/{r['total']}"
        passing_tests = ''
        failing_tests = ''
        for c in r['checks']:
            if c['status'] == 1:
                passing_tests += f"PASS {c['assert']} == {c['eq']}\n"
            else:
                failing_tests += f"FAIL {c['assert']} != {c['eq']} got {c['got']}\n"

        out[testid]['models'][id] = {
            'check_summary': check_summary,
            'passing_tests': passing_tests,
            'failing_tests': failing_tests,
            'code': r['code']
        }
        out[testid]['prompt'] += Template(model_prompt).render(id=id, passing_tests=passing_tests, failing_tests=failing_tests, **r)

    id = id + 1

import importlib  
api = importlib.import_module("interview-langchain")
params, model = api.init_model('openai/chatgpt', json.load(open('params/precise.json')))
chain = LLMChain(llm=model, prompt=PromptTemplate(template='{input}', input_variables=['input']))

for testid in out.keys():
    summary = chain.run(input=out[testid]['prompt']+footer_prompt)
    out[testid]['summary'] = summary

    print(f"----- {testid} -----")
    for id in out[testid]['models'].keys():
        print(models[id], "   ", out[testid]['models'][id]['check_summary'])
    print()
    print(summary)
    print()

with open(f'autoanalysis-orca-mini-{TEST_LANGUAGE}.json', 'w') as f:
    json.dump({ 'tests': out, 'models': models }, f, indent=2)