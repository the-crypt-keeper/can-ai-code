#!/usr/bin/env python3
import json
import os
from jinja2 import Template
import fire
import yaml

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

        models.append({'prompt': prompt, 'short_name': info['short_name'], 'params': params, 'model': model, 'id': id, 'passed': 0, 'total': 0})
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
                'code': r['code'],
                'answer': r['answer']
            }

            models[idx]['passed'] += r['passed']
            models[idx]['total'] += r['total']

    return { 'tests': out, 'models': models }

def main(config: str, path: str = "results/"):
    cfg = yaml.safe_load(open(config))
    data = prepare(cfg['language'], path, cfg['models'])
    data['config'] = cfg

    outfile = config.replace('.yaml', '.json')
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)
