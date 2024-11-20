#!/usr/bin/env python3
import argparse
import json
from time import sleep, time
from prepare import save_interview
from jinja2 import Template
import litellm
import requests
from prepare import prepare_interview

def convert_params(params):
    # integrating liteLLM to provide a standard I/O interface for every LLM
    # see https://docs.litellm.ai/docs/providers for list of supported providers
    remap = { 'max_new_tokens': 'max_tokens', 'repetition_penalty': 'presence_penalty'}
    model_params = {}
    for k,v in params.items():
        if remap.get(k): k=remap[k]
        model_params[k] = v
    return model_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interview executor for LiteLLM')
    parser.add_argument('--input', type=str, help='path to prepare*.ndjson from prepare stage')
    parser.add_argument('--interview', type=str, help='name of interview to run directly')
    parser.add_argument('--model', type=str, default='openai/chatgpt', help='model to use')
    parser.add_argument('--apibase', type=str, help='api base url override')
    parser.add_argument('--apikey', type=str, help='api key (if required)')
    parser.add_argument('--runtime', type=str, help='override runtime (when using openai-compatible server)')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use (helps determinism)')
    parser.add_argument('--params', type=str, default='params/greedy-openai.json', help='parameter file to use')
    parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
    parser.add_argument('--templateout', type=str, help='output template')
    parser.add_argument('--stop', type=str, help='stop sequences list json')
    parser.add_argument('--debug', help='enable litellm debug mode')
    args = parser.parse_args()
    
    if not (args.input or args.interview): raise Exception("You must provide one of --input or --interview.")

    # Load params and init model
    params = convert_params(json.load(open(args.params)))
    litellm.drop_params=True
    model_name = args.model
    runtime = model_name.split('/')[0]
    if args.debug: litellm.set_verbose=True
        
    # OpenAI custom base
    if args.apibase: 
        # Normalize the base, must end in /v1
        if args.apibase.endswith('/'): args.apibase = args.apibase[:-1]
        if args.apibase.endswith('/v1'): args.apibase = args.apibase[:-3]
        args.apibase += '/v1'
        params['api_base'] = args.apibase

        try:
            model_info = requests.get(args.apibase + '/models').json()
            if args.model == 'openai/chatgpt':
                model_name = 'openai/'+model_info['data'][0]['id']
            else:
                selected_model = [x for x in model_info['data'] if x['id'] == args.model.replace('openai/','')]
                if len(selected_model) == 0: raise Exception(f'Unable to find {args.model} at {args.apibase}')
                model_name = 'openai/'+selected_model[0]['id']
            args.model = model_name.split('/')[-1].replace('.gguf','')
            print('> Detected model', model_name, args.model)
        except:
            raise Exception(f'Unable to reach {args.apibase}/models')
        
        if 'koboldcpp/' in model_name:
            runtime = 'koboldcpp'
        elif model_info['data'][0].get('owned_by') == 'llamacpp':
            runtime = 'llamacpp'
        elif model_info['data'][0].get('owned_by') == 'tabbyAPI':
            runtime = 'tabbyAPI'
        elif args.runtime:
            runtime = args.runtime
        else:
            raise Exception("Unable to auto-detect, please provide --runtime if --apibase is set")
        print('> Detected runtime', runtime)

        # Set a dummy key so it doesnt complain
        if not args.apikey: args.apikey = 'xx-key-ignored'

    if args.apikey:
        params['api_key'] = args.apikey

    if args.stop:
        params['stop'] = json.loads(args.stop)
        
    # Collect interviews
    interviews = []
    if args.input:
        for input_file in args.input.split(','):
            interview = [json.loads(line) for line in open(input_file)]
            interviews.append( (input_file, interview) )
            print(f"Loaded {len(interview)} questions from {input_file}.")

    if args.interview:
        for interview_name in args.interview.split(','):
            language = "python,javascript"
            template_name = "chat-simple"            
            message_template = [{'role': 'user', 'content': Template("Write a {{language}} function {{Signature}} {{Input}} that returns {{Output}}")}]            
            output_filename, interview = prepare_interview(interview_name, language, message_template, template_name, None)
            interviews.append( (output_filename, interview) )
            print(f"Expanded {len(interview)} questions from {interview_name}.")

    # Run interviews
    output_template = Template(open(args.templateout).read()) if args.templateout else None
    for input_file, interview in interviews:
        results = []     

        for idx, challenge in enumerate(interview):
            print(f"{idx+1}/{len(interview)} {challenge['name']} {challenge['language']}")
            messages = [{'role': 'user', 'content': challenge['prompt']}]
            t0 = time()
            response = litellm.completion(model=model_name, messages=messages, seed=args.seed, **params)
            t1 = time()
            speed = response.usage.completion_tokens/(t1-t0)
            
            msg = response.choices[0].message
            answer = msg['content'] if isinstance(msg,dict) else msg.content
            answer = output_template.render(**challenge, Answer=answer) if output_template else answer            

            print()
            print(answer)
            print(f"PERF: {model_name} generated {response.usage.completion_tokens} tokens in {t1-t0:.2f}s, {speed:.2f} tok/sec")
            print()

            result = challenge.copy()
            result['answer'] = answer
            result['params'] = params
            result['model'] = args.model
            result['runtime'] = runtime

            results.append(result)

            if args.delay:
                sleep(args.delay)

        save_interview(input_file, 'none', args.params, args.model, results)
