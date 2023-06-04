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

# Load params and init model
params = json.load(open(args.params))

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

llama_command = f"{args.main} {args.args} --threads {args.threads} --model {args.model}"

for k,v in param_map.items():
    if v in params:
        llama_command += f' --{k} {params[v]}'

model_name = Path(args.model).stem

# Load interview
interview = [json.loads(line) for line in open(args.input)]
results = []

for challenge in interview:
    answer = None

    prompt_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    prompt_file.write(challenge['prompt'])
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

    answer, rv = run_shell_command(cmdline)
    if rv != 0:
        print('Failed to execute:', output)
        exit(1)

    # remove prompt from answer
    answer = answer[len(challenge['prompt']):]

    print()
    print(answer)
    print()

    result = challenge.copy()
    result['answer'] = answer
    result['params'] = { 'cmdline': cmdline }
    result['model'] = model_name

    results.append(result)

save_interview(args.input, 'none', args.params, model_name, results)