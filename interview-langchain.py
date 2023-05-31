#!/usr/bin/env python3
from langchain import LLMChain, PromptTemplate
import argparse
import json
from time import sleep
from pathlib import Path

def rekey(x,old,new):
    if old in x:
        x[new] = x[old]
        del x[old]
    return x

def adjust_params(model, params):
    if model == 'ai21/j2-jumbo-instruct':
        params = rekey(params, 'max_new_tokens', 'maxTokens')
    elif model == 'openai/chatgpt':
        params = rekey(params, 'max_new_tokens', 'max_tokens')
        params = rekey(params, 'repetition_penalty', 'presence_penalty')
        del params['top_k'] # not supported by ChatGPT
    return params

def init_model(provider, **kwargs):
    if provider == 'cohere/command-nightly':
        from langchain import Cohere
        return Cohere(model='command-nightly',**kwargs)
    elif provider == 'ai21/j2-jumbo-instruct':
        from langchain.llms import AI21
        return AI21(model='j2-jumbo-instruct', **kwargs)
    elif provider == 'openai/chatgpt':
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(model='gpt-3.5-turbo', **kwargs)
    elif provider == 'openai/gpt4':
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(model='gpt-4', **kwargs)
    raise Exception('Unsupported provider')


def prompt_template(model):
    FILENAMES = {
        'openai/chatgpt': 'prompts/openai-chatgpt.txt',
        'cohere/command-nightly': 'prompts/cohere-command-nightly.txt',
        'ai21/j2-jumbo-instruct': 'prompts/ai21-j2-jumbo-instruct.txt',
    }
    filename = FILENAMES.get(model)
    if filename:
        with open(filename) as f:
            return f.read()
    print('WARNING: Failed to load template for provider '+model)
    return '{{prompt}}' 

parser = argparse.ArgumentParser(description='Interview executor for LangChain')
parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
parser.add_argument('--model', type=str, default='openai/chatgpt', help='model to use')
parser.add_argument('--params', type=str, required=True, help='parameter file to use')
parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
args = parser.parse_args()

params = json.load(open(args.params))
for key in list(params.keys()):
    if key[0] == '$':
        del params[key]

params = adjust_params(args.model, params)
model = init_model(args.model, **params)

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
base_name = Path(args.input).stem.replace('prepare','interview')
templateout_name = 'none'
params_name = Path(args.params).stem
model_name = args.model.replace('/','-')

output_filename = 'results/'+'_'.join([base_name, templateout_name, params_name, model_name])+'.ndjson'
with open(output_filename, 'w') as f:
    f.write('\n'.join([json.dumps(result) for result in results]))
print('Saved results to', output_filename)