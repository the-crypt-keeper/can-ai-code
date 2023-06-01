#!/usr/bin/env python3
from langchain import LLMChain, PromptTemplate
import argparse
import json
import time
from time import sleep
from pathlib import Path

def rekey(x,old,new):
    if old in x:
        x[new] = x[old]
        del x[old]
    return x

def init_model(model, params):
    # LangChain did not bother to standardize the names of any of the parameters,
    # or even how to interact with them.  This is a hack to make things consistent.

    if model == 'ai21/j2-jumbo-instruct':
        from langchain.llms.ai21 import AI21PenaltyData
        from langchain.llms import AI21

        model_params = {
            'temperature': params['temperature'],
            'maxTokens': params['max_new_tokens'],
            'topP': params['top_p'],
            'presencePenalty': AI21PenaltyData()
        }
        model_params['presencePenalty'].scale = params['repetition_penalty'] - 1.0

        return model_params, AI21(model='j2-jumbo-instruct', **model_params)

    elif model == 'openai/chatgpt' or model == 'openai/gpt4':
        from langchain.chat_models import ChatOpenAI

        model_params = {
            'temperature': params['temperature'],
            'max_tokens': params['max_new_tokens'],
            'top_p': params['top_p'],
            'presence_penalty': params['repetition_penalty']
        }

        return model_params, ChatOpenAI(model_name='gpt-3.5-turbo' if model == 'openai/chatgpt' else 'gpt-4', **model_params)
    elif model == 'cohere/command-nightly':
        from langchain import Cohere

        model_params = {
            'temperature': params['temperature'],
            'max_tokens': params['max_new_tokens'],
            'p': params['top_p'],
            'k': params['top_k'],
            'frequency_penalty': params['repetition_penalty'] - 1.0
        }

        return model_params, Cohere(model='command-nightly', **model_params)
    
    raise Exception('Unsupported model/provider')

parser = argparse.ArgumentParser(description='Interview executor for LangChain')
parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
parser.add_argument('--model', type=str, default='openai/chatgpt', help='model to use')
parser.add_argument('--params', type=str, required=True, help='parameter file to use')
parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
args = parser.parse_args()

# Load params and init model
params, model = init_model(args.model, json.load(open(args.params)))

# Load interview
interview = [json.loads(line) for line in open(args.input)]
results = []

for challenge in interview:
    chain = LLMChain(llm=model, prompt=PromptTemplate(template='{input}', input_variables=['input']))
    answer = chain.run(input=challenge['prompt'])

    print()
    print(answer)
    print()

    result = challenge.copy()
    result['answer'] = answer
    result['params'] = params
    result['model'] = args.model

    results.append(result)

    if args.delay:
        sleep(args.delay)

# Save results
[stage, interview_name, languages, template, *stuff] = Path(args.input).stem.split('_')
templateout_name = 'none'
params_name = Path(args.params).stem
model_name = args.model.replace('/','-')
ts = str(int(time.time()))

output_filename = 'results/'+'_'.join(['interview', interview_name, languages, template, templateout_name, params_name, model_name, ts])+'.ndjson'
with open(output_filename, 'w') as f:
    f.write('\n'.join([json.dumps(result, default=vars) for result in results]))
print('Saved results to', output_filename)