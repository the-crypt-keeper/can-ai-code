#!/usr/bin/env python3
import argparse
import json
from time import sleep, time
from prepare import save_interview
from jinja2 import Template
import litellm
import requests
from prepare import cli_to_interviews
import concurrent.futures

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
    parser.add_argument('--interview', type=str, default='senior', help='name of interview to run directly')
    parser.add_argument('--prompt', type=str, help='chat template for interview', default='prompts/chat.json')
    parser.add_argument('--model', type=str, default='openai/chatgpt', help='model to use')
    parser.add_argument('--apibase', type=str, help='api base url override')
    parser.add_argument('--apikey', type=str, help='api key (if required)')
    parser.add_argument('--runtime', type=str, help='override runtime (when using openai-compatible server)')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use (helps determinism)')
    parser.add_argument('--params', type=str, default='params/greedy-openai.json', help='parameter file to use')
    parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
    parser.add_argument('--context', type=int, default=0, help='override context size (max_tokens)')
    parser.add_argument('--templateout', type=str, help='output template')
    parser.add_argument('--stop', type=str, help='stop sequences list json')
    parser.add_argument('--debug', help='enable litellm debug mode', action='store_true')
    parser.add_argument('--parallel', type=int, default=0, help='number of parallel completions to run (0 for sequential)')
    args = parser.parse_args()
    
    if not (args.input or args.interview): raise Exception("You must provide one of --input or --interview.")

    # Load params and init model
    params = convert_params(json.load(open(args.params)))
    if args.context > 0: params['max_tokens'] = args.context
    litellm.drop_params=True
    model_name = args.model
    runtime = model_name.split('/')[0]
    if args.debug: litellm.set_verbose=True
        
    # OpenAI custom base
    if args.apibase:
        if 'ollama' not in args.model:
            # Normalize the base, must end in /v1
            if args.apibase.endswith('/'): args.apibase = args.apibase[:-1]
            if args.apibase.endswith('/v1'): args.apibase = args.apibase[:-3]
            args.apibase += '/v1'            
        params['api_base'] = args.apibase

    if args.apibase and 'ollama' not in args.model:
        try:
            target_model = args.model.replace('openai/','').replace('text-completion-openai/','').replace('text/','')
            model_info = requests.get(args.apibase + '/models').json()
            if args.model == 'openai/chatgpt':
                model_name = 'openai/'+model_info['data'][0]['id']
            else:
                selected_model = [x for x in model_info['data'] if x['id'] == target_model]
                if len(selected_model) == 0: raise Exception(f'Unable to find {args.model} at {args.apibase}')
                if 'text-completion-openai/' in args.model or 'text/' in args.model:
                    model_name = 'text-completion-openai/'+selected_model[0]['id']
                else:
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
        
    def process_challenge(challenge, model_name, params, seed, output_template):
        print(f"Processing: {challenge['name']} {challenge['language']}")
        messages = challenge['prompt']
        if isinstance(messages, str):
            print('WARNING: Using text completion.')
            messages = [{'role': 'user', 'content': messages}]

        t0 = time()
        response = litellm.completion(model=model_name, messages=messages, seed=seed, timeout=3600, **params)
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

        return result

    # Run interviews
    interviews = cli_to_interviews(args.input, args.interview, None, args.prompt)
    output_template = Template(open(args.templateout).read()) if args.templateout else None
    
    for input_file, interview in interviews:
        results = []
        
        if args.parallel > 0:
            # Run challenges in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = []
                for challenge in interview:
                    futures.append(executor.submit(
                        process_challenge, 
                        challenge, 
                        model_name, 
                        params, 
                        args.seed, 
                        output_template
                    ))
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if args.delay:
                        sleep(args.delay)
        else:
            # Run challenges sequentially
            for idx, challenge in enumerate(interview):
                print(f"{idx+1}/{len(interview)} {challenge['name']} {challenge['language']}")
                result = process_challenge(challenge, model_name, params, args.seed, output_template)
                results.append(result)
                if args.delay:
                    sleep(args.delay)

        save_interview(input_file, 'none', args.params, args.model, results)
