#!/usr/bin/env python3
import json
import os
from jinja2 import Template
import fire
import yaml
from copy import copy

task_prompt = "Write a {{language}} function {{Signature}} {{Input}} that returns {{Output}}"

def prepare(TEST_LANGUAGE, path, files):
    out = {}
    models = []

    for idx, info in enumerate(files):
        file = os.path.join(path, info['eval'])
        id = info['id']

        tags = os.path.basename(file).replace('.ndjson', '').split('_')
        prompt = tags[3]
        params = tags[5]
        model = tags[6]

        models.append({'prompt': prompt, 'short_name': info.get('short_name',id), 'params': params, 'model': model, 'id': id, 'idx': idx, 'passed': 0, 'total': 0})
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
                    eq = "inside" if 'eq-any' in c else '=='
                    passing_tests += f"PASS {c['assert']} {eq} {c.get('eq',c.get('eq-any'))}\n"
                else:
                    neq = "not inside" if 'eq-any' in c else '!='
                    failing_tests += f"FAIL {c['assert']} {eq} {c.get('eq',c.get('eq-any'))} got {c['got']}\n"

            out[testid]['results'][id] = {
                'check_summary': check_summary,
                'passing_tests': passing_tests,
                'failing_tests': failing_tests,
                'code': r['code'],
                'answer': r['answer']
            }

            models[idx]['passed'] += r['passed']
            models[idx]['total'] += r['total']

    return { 'tests': out, 'models': models }

header_prompt = """
You are going to evaluate the results of language models on a {{language}} programming challenge: {{task}}
Automated tests have been used to verify corectness each solution produced, a detailed description of the results of each test will be provided.
For each model, you will be provided the code produced by the model and the result of all tests.
Compare and contrast the solutions each model produced.  Do not repeat any of the generated code back to me.  Highlight differences in solution approaches, test results, and provide a final summary of cohort performance on this challenge.

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

def analysis(data, analyser):
    from langchain.chat_models import ChatOpenAI
    from langchain import LLMChain, PromptTemplate

    params = json.load(open('params/precise.json'))
    model_params = {
        'temperature': params['temperature'],
        'max_tokens': params['max_new_tokens'],
        'top_p': params['top_p'],
        'presence_penalty': params['repetition_penalty']
    }

    model = ChatOpenAI(model_name=analyser, **model_params)
    chain = LLMChain(llm=model, prompt=PromptTemplate(template='{input}', input_variables=['input']))

    models = {}
    for idx, model_info in enumerate(data['models']):
        models[model_info['id']] = model_info

    out = data['tests']
    for testid in out.keys():

        print(f"----- {testid} -----")
        prompt = Template(header_prompt).render(**out[testid])
        for idx in out[testid]['results'].keys():
            model_info = models[idx]
            print(model_info, "   ", out[testid]['results'][idx]['check_summary'])
            prompt += Template(model_prompt).render(**out[testid]['results'][idx], id=model_info['id'])
        prompt += Template(footer_prompt).render(**out[testid])

        out[testid]['summary'] = chain.run(input=prompt)

        print()
        print(out[testid]['summary'])
        print()

    return data

def main(config: str, path: str = "results/", analyser: str = "", language: str = "javascript,python"):
    cfg = yaml.safe_load(open(config))

    for lang in language.split(','):
        cfg['language'] = lang
        print('Comparing results for', lang)
        data = prepare(cfg['language'], path, cfg['models'])
        data['config'] = copy(cfg)
        data['config']['title'] += f" ({lang})"
        data['analyser'] = analyser

        if analyser != "":
            analysis(data, analyser)

        outfile = config.replace('.yaml', f'-{lang}.json')
        with open(outfile, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)
