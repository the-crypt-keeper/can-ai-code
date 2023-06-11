#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from jinja2 import Template
import sys
sys.path.append('..')
from evaluate import extract_code
from evalplus.data import get_human_eval_plus, get_human_eval, write_jsonl

def prepare_humaneval(args):
    template = Template(open(args.template).read())
    template_name = Path(args.template).stem

    args.language = "python"
    args.interview = "evalplus" if args.plus else "humaneval" 
    questions = get_human_eval_plus() if args.plus else get_human_eval()

    output_filename = f"prepare_{args.interview}_{args.language.replace(',', '-')}_{template_name}.ndjson"
    outputs = []
    for task_id in questions:
            test = questions[task_id]
            test['name'] = test['task_id']
            prompt = template.render(test)
           
            output = {
                'name': test['task_id'],
                'task_id': test['task_id'],
                'entry_point': test['entry_point'],
                'language': args.language,
                'prompt': prompt,
            }
            outputs.append(output)

    write_jsonl(output_filename, outputs)
    print(f"Expanded {len(outputs)} {template_name} prompts to {output_filename}")

def remove_lines_until_def(input_string):
    lines = input_string.split('\n')  # Split the input string into a list of lines
    imports = []
    for index, line in enumerate(lines):
        if 'import' in line:
            imports.append(line.strip())
        elif 'def' in line:
            first_line_indent = len(lines[index+1]) - len(lines[index+1].lstrip())
            imports = [' '*first_line_indent + line for line in imports]
            print(imports)

            return '\n'.join(imports+lines[index+1:])  # Join the remaining lines and return as a string
    return ''  # Return an empty string if no line with 'def' is found

def format_humaneval(args):
    results = [json.loads(line) for line in open(args.answers)]

    outfile = args.answers.replace('.ndjson', '.jsonl')
    with open(outfile, 'w') as file:
         for result in results:
             completion = extract_code(result['answer'])
             cleaned = remove_lines_until_def(completion)
             file.write(json.dumps({"task_id": result["task_id"], "completion": cleaned}) + '\n')

    print(f"Formatted answers to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interview preparation')
    parser.add_argument('--template', type=str, help='prepare interview')
    parser.add_argument('--plus', action='store_true', help='use humanevalplus instead of humaneval')
    parser.add_argument('--answers', type=str, help='post-process results')
    args = parser.parse_args()

    if args.template:
        prepare_humaneval(args)
    elif args.answers:
        format_humaneval(args)
    else:
        parser.print_help()
        exit(1)