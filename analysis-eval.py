from jinja2 import Template
from langchain import LLMChain, PromptTemplate
import json
import sys

header_prompt = """
You are going to evaluate the results of language models on a {{language}} programming challenge: {{task}}
Automated tests have been used to verify corectness each solution produced, a detailed description of the results of each test will be provided.
For each model, you will be provided the code produced by the model and the result of all tests.
Compare and constract the solutions each model produced.  Do not repeat the code back to me.  Highlight differences in test results, and provide a final summary of cohort performance on this challenge.

"""

model_prompt = """
---
Model: {{id}}
Test Result: {{check_summary}}
Test Details:
{{passing_tests}}{{failing_tests}}
Code:
```{{language}}
{{code}}
```
"""

footer_prompt = """
---
Analysis:"""

import importlib  
api = importlib.import_module("interview-langchain")
params, model = api.init_model('openai/chatgpt', json.load(open('params/precise.json')))
chain = LLMChain(llm=model, prompt=PromptTemplate(template='{input}', input_variables=['input']))

data = json.load(open(sys.argv[1]))
out = data['tests']
models = data['models']

aliases = {
    'jondurbin-airoboros-13b-gpt4-1.4-fp16': 'FP16',
    'airoboros-13b-gpt4-1.4.ggmlv3.q5-0': 'GGML-q5_0',
    'TheBloke-airoboros-13B-gpt4-1.4-GPTQ': 'GPTQ-4b'
}

for testid in out.keys():

    print(f"----- {testid} -----")
    prompt = Template(header_prompt).render(**out[testid])
    for idx in out[testid]['results'].keys():
        model_info = models[int(idx)]
        print(model_info, "   ", out[testid]['results'][idx]['check_summary'])
        prompt += Template(model_prompt).render(**out[testid]['results'][idx], id=aliases[model_info['model']])
    prompt += Template(footer_prompt).render(**out[testid])

    out[testid]['summary'] = chain.run(input=prompt)

    print()
    print(out[testid]['summary'])
    print()

print(json.dumps(data, indent=2), file=sys.stderr)