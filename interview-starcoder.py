#!/usr/bin/env python3

# pip install -q transformers==4.29.2
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template
import argparse
import json
from prepare import save_interview

parser = argparse.ArgumentParser(description='Interview executor for StarCoder family')
parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
parser.add_argument('--model', type=str, default='bigcode/tiny_starcoder_py', help='model to use')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
parser.add_argument('--templateout', type=str, required=True, help='output template file')
parser.add_argument('--params', type=str, required=True, help='parameter file to use')
args = parser.parse_args()

# Load model
checkpoint = args.model
device = args.device

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Load parameters and output template, strip json comments.
params = json.load(open(args.params))
for key in list(params.keys()):
    if key[0] == '$':
        del params[key]
output_template = Template(open(args.templateout).read())

# Load interview
interview = [json.loads(line) for line in open(args.input)]
results = []
for challenge in interview:

    prompt = challenge['prompt']
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)

    result = tokenizer.decode(outputs[0])
    result = result.replace(prompt, '').replace('<|endoftext|>','')
    
    answer = output_template.render(**challenge, Answer=result)

    result = challenge.copy()
    result['answer'] = answer
    result['params'] = params
    result['model'] = args.model
    result['runtime'] = 'transformers'

    results.append(result)

    print()
    print(answer)
    print()

save_interview(args.input, args.templateout, args.params, args.model, results)