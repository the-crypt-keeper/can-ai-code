#!/usr/bin/env python3
import glob
import yaml
import argparse
import json
from jinja2 import Template
from pathlib import Path

def load_questions(interview='junior-dev'):
    for file_path in glob.glob(interview+'/*.yaml'):
        with open(file_path, 'r') as file:
            tests = yaml.safe_load(file)
            for test in tests.keys():
                if test[0] == '.':
                    continue
                tests[test]['name'] = test
                yield tests[test]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interview preparation')
    parser.add_argument('--language', type=str, default='python,javascript', help='languages to prepare, comma seperated')
    parser.add_argument('--interview', type=str, default='junior-dev', help='interview to prepare')
    parser.add_argument('--template', type=str, required=True, help='prompt template file')
    args = parser.parse_args()

    template = Template(open(args.template).read())
    template_name = Path(args.template).stem

    output_filename = f"results/prepare_{args.interview}_{args.language.replace(',', '-')}_{template_name}.ndjson"
    outputs = []
    for test in load_questions():
        for language in args.language.split(','):
            prompt = template.render({'language': language, **test})
            
            output = test.copy()
            del output['Checks']
            output['language'] = language
            output['prompt'] = prompt
            outputs.append(output)

    with open(output_filename, 'w') as file:
        file.write('\n'.join([json.dumps(output) for output in outputs]))
        print(f"Expanded {len(outputs)} {template_name} prompts to {output_filename}")