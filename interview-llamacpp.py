#!/usr/bin/env python3
import argparse
import json
import tempfile
from sys import exit
from pathlib import Path
from sbox.sandbox import run_shell_command
from prepare import save_interview

parser = argparse.ArgumentParser(description='Interview executor for LlamaCpp')
parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
parser.add_argument('--model', type=str, required=True, help='path to model file to use')
parser.add_argument('--params', type=str, required=True, help='parameter file to use')

parser.add_argument('--main', type=str, default='~/ai/latest/main', help='path to llama.cpp main binary')
parser.add_argument('--threads', type=int, default=4, help='number of threads to use')
parser.add_argument('--args', type=str, default='--ctx_size 2048 --batch_size 1024', help='misc arguments to pass to main (ex gpu offload)')
parser.add_argument('--ssh', type=str, help='(optional) ssh hostname for remote execution')
args = parser.parse_args()

def build_llama_command(args, params):
    param_map = {
        'n_predict': 'max_new_tokens',
        'temp': 'temperature',
        'top_k': 'top_k',
        'top_p': 'top_p',
        'repeat_last_n': 'repeat_last_n',
        'repeat_penalty': 'repetition_penalty',
        'mirostat': 'mirostat',
        'mirostat-lr': 'mirostat-lr',
        'mirostat-ent': 'mirostat-ent'
    }

    if args.main.find('starcoder') > -1:
        param_map['repeat-penalty'] = 'repetition_penalty'
        param_map['repeat-last-n'] = 'repeat_last_n'
        del param_map['repeat_last_n']
        del param_map['repeat_penalty']

    llama_command = f"{args.main} {args.args} --threads {args.threads} --model {args.model}"

    for k,v in param_map.items():
        if v in params:
            llama_command += f' --{k} {params[v]}'

    return llama_command

model_name = Path(args.model).stem

tasks = []
for param_file in args.params.split(','):
    for input_file in args.input.split(','):
        tasks.append((param_file, input_file))

for param_file, input_file in tasks:
    print('Executing',args.model,'with parameter file',param_file,'and input file',input_file)
    
    params_json = json.load(open(param_file))
    llama_command = build_llama_command(args, params_json)
    interview = [json.loads(line) for line in open(input_file)]
    results = []
    for idx, question in enumerate(interview):
        answer = None

        print(f"[{idx+1}/{len(interview)}] {question['language']} {question['name']}")

        prompt_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        prompt_file.write(question['prompt'])
        prompt_file.close()

        cmdline = llama_command + f' --file {prompt_file.name}'

        if args.ssh:
            scp_command = f"scp {prompt_file.name} {args.ssh}:{prompt_file.name}"
            print('Copying to remote machine:', scp_command)

            output, rv = run_shell_command(scp_command)
            if rv != 0:
                print('Failed to copy to remote machine:', output)
                exit(1)

            cmdline = f"ssh {args.ssh} '{cmdline}'"

        print('Executing llama.cpp: '+cmdline)

        answer, rv = run_shell_command(cmdline, stdout_only=True)
        if rv != 0:
            print('Failed to execute:', output)
            exit(1)

        # remove prompt from answer
        start_offset = max(answer.rfind(question['prompt']), 0)
        start_offset += len(question['prompt'])
        answer = answer[start_offset:]

        # for starcoder remove the trailer
        end_offset = answer.find('\nmain: mem per token =')
        if end_offset > -1:
            answer = answer[:end_offset]

        # remove any eos/eot tokens
        for eos in ['<|endoftext|>', '<|end|>', '<|end_of_turn|>']:
            answer = answer.replace(eos, '')

        print()
        print(answer)
        print()

        result = question.copy()
        result['answer'] = answer
        result['params'] = { 'cmdline': cmdline }
        result['model'] = model_name
        result['runtime'] = 'llamacpp'

        results.append(result)

    save_interview(input_file, 'none', param_file, model_name, results)