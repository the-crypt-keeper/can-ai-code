#!/usr/bin/env python3
from gradio_client import Client
from bs4 import BeautifulSoup
import time
import json
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Interview executor for StarChat on a Huggingface Space')
parser.add_argument('--questions', type=str, required=True, help='path to questions .csv from prepare stage')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir) 

def remove_indentation(code_block):
    lines = code_block.split('\n')
    if not lines:
        return code_block
    
    first_line_indent = len(lines[0]) - len(lines[0].lstrip())
    modified_lines = [line[first_line_indent:] for line in lines]
    modified_code = '\n'.join(modified_lines)
    return modified_code

def run_starchat(prompt):
	# If you dont re-init the client, it starts a chat with history.
	client = Client("https://HuggingFaceH4-starchat-playground.hf.space/")

	# Introspection API to see what parameters are available
	#client.view_api()

	with open("/tmp/json_empty",'w') as f:
		json.dump([], f)

	job = client.submit(
			"All code should be contained in ``` blocks.",
			prompt,
			"/tmp/json_empty",
			0.2, # temperature
			50,  # top_k
			0.95, # top_p
			512, # max_tokens
			1.2, # repeat_penalty
			False,
			fn_index=2
	)
	while not job.done():
		time.sleep(0.5)
		print(job.status())

	outputs = job.outputs()

	with open(outputs[-1][0], 'r') as f:
		answer = json.load(f)

	soup = BeautifulSoup(answer[0][1], "html.parser")

	longest_code = ""
	for item in soup.find_all('code'):
		if len(item.get_text()) > len(longest_code):
			print("Found candidate code: ", item)
			longest_code = remove_indentation(item.get_text())

	return answer[0][1] if longest_code == "" else longest_code

df = pd.read_csv(args.questions)
for idx, test in df.iterrows():
    print(test['name'])
    out_file = args.outdir+'/'+test['name']+'.txt'

    if os.path.exists(out_file):
        print('Skipping, already exists')
        continue
    
    answer = run_starchat(test['prompt'])
    print(answer)

    with open(out_file, 'w') as f:
    	f.write(answer)