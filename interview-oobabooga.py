#!/usr/bin/env python3
import json
import argparse
import requests
from prepare import save_interview

parser = argparse.ArgumentParser(description='Interview executor for text-generation-web-ui or KoboldCpp API v1 compatible server')
parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
parser.add_argument('--params', type=str, required=True, help='parameter file to use')
parser.add_argument('--model', type=str, required=True, help='model name being evaluated')
parser.add_argument('--host', type=str, default='localhost:5000', help="host to connect to")
parser.add_argument('--kobold', action='store_true', help='use koboldcpp server instead of text-generation-web-ui')
args = parser.parse_args()

URL_TAIL='/api/v1/generate'
URI = f'{args.host}{URL_TAIL}' if '://' in args.host else f'http://{args.host}{URL_TAIL}'

def kobold_params(params):
    # Params list: https://github.com/LostRuins/koboldcpp/blob/concedo/koboldcpp.py#L326
    return {
        'max_length': params['max_new_tokens'],
        'temperature': params['temperature'],
        'top_k': params['top_k'],
        'top_p': params['top_p'],
        'rep_pen': params['repetition_penalty'],
        'rep_pen_range': params.get('repeat_last_n', 128)
    }
raw_params = json.load(open(args.params))
params = kobold_params(raw_params) if args.kobold else raw_params

def send_request(prompt):
    request = params.copy()
    request["prompt"] = prompt
    response = requests.post(URI, json=request)    
    assert response.status_code == 200, response.content
    result = response.json()['results'][0]['text']
    return result 

def run():
    interview = [json.loads(line) for line in open(args.input)]
    results = []

    if args.kobold:
        print('### WARNING ###')
        print('kobold server must be launched with --unbantokens for code generation to work.')
        print('### WARNING ###')
        print()

    for test in interview:
        print(test['name'])
        
        answer = send_request(test['prompt'])

        print(answer)

        result = test.copy()
        result['answer'] = answer
        result['params'] = params
        result['model'] = args.model
        result['runtime'] = 'api-oobabooga'

        results.append(result)
    
    save_interview(args.input, 'none', args.params, args.model, results)

if __name__ == "__main__":
    run()
