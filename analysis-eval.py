from jinja2 import Template
from langchain import LLMChain, PromptTemplate
import json
import sys

header_prompt = """
You are going to evaluate the results of language models on a {{language}} programming challenge.
The challenge given to each model is to {{task}}.
You will be provided the code produced by each model.
Automated tests have evaluated the performance of each model on the challenge, a list of passing and failing tests will also be provided.
Compare and constract the solutions each model produced, highlighting any differences in test results and provide a final summary of the results.
"""

model_prompt = """---
Model: {{id}}
Test Result: {{check_summary}}
Test Details:
{{passing_tests}}{{failing_tests}}
Code:
```{{language}}
{{code}}
```
"""

footer_prompt = """---

Analysis:"""

import importlib  
api = importlib.import_module("interview-langchain")
params, model = api.init_model('openai/chatgpt', json.load(open('params/precise.json')))
chain = LLMChain(llm=model, prompt=PromptTemplate(template='{input}', input_variables=['input']))

data = json.load(open(sys.argv[1]))
out = data['tests']
models = data['models']

for testid in out.keys():

    print(f"----- {testid} -----")
    prompt = Template(header_prompt).render(**out[testid])
    for id in out[testid]['models'].keys():
        print(models[id], "   ", out[testid]['models'][id]['check_summary'])
        prompt += Template(model_prompt).render(**out[testid]['models'][id])
    prompt += Template(footer_prompt).render(**out[testid])

    out[testid]['summary'] = chain.run(input=prompt)

    print()
    print(out[testid]['summary'])
    print()

print(json.dumps(out, indent=2))