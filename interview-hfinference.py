#!/usr/bin/env python3
import json
import requests
import os
import time
from jinja2 import Template
import argparse
from pathlib import Path
from prepare import save_interview

parser = argparse.ArgumentParser(description='Interview executor for HuggingFace Inference API')
parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
parser.add_argument('--model', type=str, default='bigcode/starcoder', help='model to use')
parser.add_argument('--params', type=str, required=True, help='parameter file to use')
parser.add_argument('--templateout', type=str, required=True, help='output template file')
args = parser.parse_args()

headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
API_URL = "https://api-inference.huggingface.co/models/"+args.model
def query(payload):
    tries = 0
    while tries < 5:
        tries += 1
        response = requests.request("POST", API_URL, headers=headers, json=payload)
        res = {}
        try:
            res = json.loads(response.content.decode("utf-8"))
        except Exception as e:
            print('JSON decoder failed:', response.content.decode("utf-8"))
            time.sleep(1)
            continue
        if not isinstance(res, list):
            try:
                print('Generation error:', res['error'])
            except:
                print('Something weird went wrong', res)
            time.sleep(1)
            continue
    return res

# Load params and adapt to model format
# see https://huggingface.github.io/text-generation-inference/ GenerateParameters struct
params = json.load(open(args.params))
model_params = {
    "temperature": params['temperature'],
    "top_k": params['top_k'],
    "top_p": params['top_p'],
    "max_new_tokens": params['max_new_tokens'],
    "repetition_penalty": params['repetition_penalty']
}

# Output template
output_template = Template(open(args.templateout).read())

# Run Interview
interview = [json.loads(line) for line in open(args.input)]
results = []

for challenge in interview:
    data = query(
        {
            "inputs": challenge['prompt'],
            "parameters": model_params,
        }
    )

    result = data[0]['generated_text']

    result = result.replace(challenge['prompt'], '').replace('<|endoftext|>','')
    
    output = output_template.render(**challenge, Answer=result)

    print()
    print(output)
    print()

    result = challenge.copy()
    result['answer'] = output
    result['params'] = model_params
    result['model'] = args.model
    result['runtime'] = 'api-inference'
    results.append(result)

# Save results
save_interview(args.input, args.templateout, args.params, args.model, results)
