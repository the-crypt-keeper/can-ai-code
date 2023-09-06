#!/usr/bin/env python3
import time
import json
import argparse
from prepare import save_interview
from gradio_client import Client

# helper for gradio spaces that require chat history to be in a file
def empty_json():
	with open("/tmp/json_empty",'w') as f:
		json.dump([], f)

def build_starchat(prompt, params, **kwargs):
	"""
	 - predict(parameter_39, select_a_model, parameter_14, parameter_22, chat, temperature, topk, topp_nucleus_sampling, max_new_tokens, repetition_penalty, store_data, fn_index=2) -> (chat, value_22)
    Parameters:
     - [Checkbox] parameter_39: bool 
     - [Radio] select_a_model: str 
     - [Textbox] parameter_14: str 
     - [Textbox] parameter_22: str 
     - [Chatbot] chat: str (filepath to JSON file) 
     - [Slider] temperature: int | float (numeric value between 0.0 and 1.0) 
     - [Slider] topk: int | float (numeric value between 0.0 and 100) 
     - [Slider] topp_nucleus_sampling: int | float (numeric value between 0.0 and 1) 
     - [Slider] max_new_tokens: int | float (numeric value between 0 and 1024) 
     - [Slider] repetition_penalty: int | float (numeric value between 0.0 and 10) 
     - [Checkbox] store_data: bool 
    Returns:
     - [Chatbot] chat: str (filepath to JSON file) 
     - [Textbox] value_22: str 
	"""
	empty_json()

	return [
		False,
		kwargs['model'],
		"",
		prompt,
		"/tmp/json_empty",
		params['temperature'], # temperature
		params['top_k'],  # top_k
		params['top_p'], # top_p
		params['max_new_tokens'], # max_tokens
		params['repetition_penalty'], # repeat_penalty
		False
	], 2

def parse_starchat(outputs, **kwargs):
	if len(outputs) == 0:
		raise Exception("No outputs - model call failed")
	with open(outputs[-1][0], 'r') as f:
		answer = json.load(f)
	return answer[0][1]

def build_wizardcoder(prompt, params, **kwargs):
	return [
		prompt,
		params['temperature'], # temperature
		params['top_p'], # top_p
		params['top_k'],  # top_k
		params.get('beams', 1), # beams
		params['max_new_tokens'], # max_tokens
	], 0

def parse_wizardcoder(outputs, **kwargs):
	return None if len(outputs) == 0 else outputs[-1]

def build_falcon(prompt, params, **kwargs):
	"""
	 - predict(message, optional_system_prompt, temperature, max_new_tokens, topp_nucleus_sampling, repetition_penalty, api_name="/chat") -> message
    Parameters:
     - [Textbox] message: str 
     - [Textbox] optional_system_prompt: str 
     - [Slider] temperature: int | float (numeric value between 0.0 and 1.0) 
     - [Slider] max_new_tokens: int | float (numeric value between 0 and 8192) 
     - [Slider] topp_nucleus_sampling: int | float (numeric value between 0.0 and 1) 
     - [Slider] repetition_penalty: int | float (numeric value between 1.0 and 2.0) 
    Returns:
     - [Textbox] message: str 
	"""
	return [
		kwargs.get('system',''),
		prompt,
		params['temperature'], # temperature
		params['max_new_tokens'], # max_tokens
		params['top_p'],
		params.get('repetition_penalty',1.0)
	], "/chat"

configs = {
	'starchat-alpha': {
		'url': 'https://HuggingFaceH4-starchat-playground.hf.space/',
		'builder': (build_starchat, { 'model': 'starchat-alpha' }),
		'parser': (parse_starchat, {})
	},
	'starchat-beta': {
		'url': '',
		'builder': (build_starchat, { 'model': 'starchat-beta' }),
		'parser': (parse_starchat, {})
	},
	'wizardcoder': {
		'url': 'https://e5eaf7d09cc1521c.gradio.app/',
		'builder': (build_wizardcoder, {}),
		'parser': (parse_wizardcoder, {})
	},
	'falcon-180b': {
		'url': 'https://tiiuae-falcon-180b-demo.hf.space/',
		'builder': (build_falcon, {}),
		'parser': (parse_wizardcoder, {})
	}
}

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Interview executor via Gradio endpoints')
	parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
	parser.add_argument('--params', type=str, required=True, help='parameter file to use')
	parser.add_argument('--config', type=str, required=True, help='one of: starchat-alpha, starchat-beta, wizardcoder')
	args = parser.parse_args()

	model_name = args.config+'-gradio'
	try:
		config = configs[args.config]
	except KeyError:
		print(f"Invalid config: {args.config}")
		print("Select one of",", ".join(configs.keys()))
		exit(1) 

	#client = Client(config['url'])
	#print(client.view_api(all_endpoints=True))
	#exit(0)
	
	raw_params = json.load(open(args.params))
	interview = [json.loads(line) for line in open(args.input)]
	results = []

	for idx, test in enumerate(interview):
		print(f"{idx+1}/{len(interview)} {test['language']} {test['name']}")

		payload, fn_index = config['builder'][0](test['prompt'], raw_params, **config['builder'][1])

		while True:
			client = Client(config['url'])
			if isinstance(fn_index, int):
				job = client.submit(*payload, fn_index=fn_index)
			else:
				job = client.submit(*payload, api_name=fn_index)

			while not job.done():
				time.sleep(2)
				print(job.status())
			
			answer = config['parser'][0](job.outputs(), **config['parser'][1])
			if answer is not None:
				break
			else:
				print('ERROR: No answer provided by model, trying again.')

		print()
		print(answer)
		print()

		result = test.copy()

		result['answer'] = answer
		result['params'] = raw_params
		result['model'] = model_name
		result['runtime'] = 'gradio'

		results.append(result)

	save_interview(args.input, 'none', args.params, model_name, results)