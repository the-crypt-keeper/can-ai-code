#!/usr/bin/env python3
from gradio_client import Client
from bs4 import BeautifulSoup
import time
import json
import argparse
import os
from urllib.parse import urlparse
from prepare import save_interview

parser = argparse.ArgumentParser(description='Interview executor for Huggingface Space (focued on StarChat)')
parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
parser.add_argument('--params', type=str, required=True, help='parameter file to use')
parser.add_argument('--endpoint', type=str, required=False, default='https://HuggingFaceH4-starchat-playground.hf.space/', help='hf space url')
parser.add_argument('--model', type=str, default='starchat-alpha', help='starchat model parameter')
parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
args = parser.parse_args()

def call_starchat(prompt, params, endpoint):
	# If you dont re-init the client, it starts a chat with history.
	client = Client(endpoint)

	# Introspection API to see what parameters are available
	#client.view_api()

	with open("/tmp/json_empty",'w') as f:
		json.dump([], f)

	job = client.submit(
			args.model,
			"",
			prompt,
			"/tmp/json_empty",
			params['temperature'], # temperature
			params['top_k'],  # top_k
			params['top_p'], # top_p
			params['max_new_tokens'], # max_tokens
			params['repetition_penalty'], # repeat_penalty
			False,
			fn_index=2
	)
	while not job.done():
		time.sleep(2)
		print(job.status())

	outputs = job.outputs()

	with open(outputs[-1][0], 'r') as f:
		answer = json.load(f)

	return answer[0][1]

model_name = urlparse(args.endpoint).netloc.replace('.','-')
raw_params = json.load(open(args.params))

interview = [json.loads(line) for line in open(args.input)]
results = []

for test in interview:
    print(test['name'])
    
    answer = call_starchat(test['prompt'], raw_params, args.endpoint)
    
    print(answer)
    
    result = test.copy()
	
    result['answer'] = answer
    result['params'] = raw_params
    result['model'] = model_name+'-'+args.model
    result['runtime'] = 'api-spaces'

    results.append(result)

save_interview(args.input, 'none', args.params, model_name, results)