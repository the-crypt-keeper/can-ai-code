#!/usr/bin/env python3
import json
import argparse
import pandas as pd
import os
import requests
from pathlib import Path
from jinja2 import Template

parser = argparse.ArgumentParser(description='Interview executor for Kobold API v1 compatible server')
parser.add_argument('--questions', type=str, required=True, help='path to questions .csv from prepare stage')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
parser.add_argument('--prompt', type=str, required=True, help="prompt template")
parser.add_argument('--debug', action='store_true', help="store sent requests alongside responses")
parser.add_argument('--host', type=str, default='localhost:5000', help="host to connect to")
parser.add_argument('--config', type=str,  default='model_parameters/default.json', help=".json file with model parameters")
args = parser.parse_args()

URL_TAIL='/api/v1/generate'
URI = f'{args.host}{URL_TAIL}' if '://' in args.host else f'http://{args.host}{URL_TAIL}'

def send_request(prompt):
    with open(args.config) as f:
        request = json.load(f)
    request["prompt"] = prompt
    response = requests.post(URI, json=request)    
    assert response.status_code == 200, response.content
    result = response.json()['results'][0]['text']
    if args.debug:
        print(json.dumps(request, indent=2))
    return request, result

def run():
    Path(args.outdir).mkdir(exist_ok=True, parents=True)

    with open(args.prompt) as f:
        prompt_template = Template(f.read())

    comment = {
        'python': '#',
        'javascript': '//'
    }
    function_prefix = {
        'python': 'def',
        'javascript': 'function'
    }

    df = pd.read_csv(args.questions)
    for idx, test in df.iterrows():
        print(test['name'])
        out_file = args.outdir+'/'+test['name']+'.txt'

        if os.path.exists(out_file):
            print('Skipping, already exists')
            continue

        full_prompt = prompt_template.render(
                prompt=test['prompt'],
                language=test['language'],
                comment=comment[test['language']],
                function_prefix=comment[test['language']]
        )
        request, answer = send_request(full_prompt)
        print(answer)

        with open(out_file, 'w') as f:
            f.write(answer)

        if args.debug:
            with open(f'{out_file}.request.json', 'w') as f:
                f.write(json.dumps(request))

if __name__ == "__main__":
    run()
