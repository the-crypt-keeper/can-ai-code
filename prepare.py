#!/usr/bin/env python3
import glob
import yaml
import argparse
import json
import time
import os
from jinja2 import Template
from pathlib import Path

def load_questions(interview='junior-v2'):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    for file_path in glob.glob(module_dir+'/'+interview+'/*.yaml'):
        with open(file_path, 'r') as file:
            tests = yaml.safe_load(file)
            for test in tests.keys():
                if test[0] == '.':
                    continue
                tests[test]['name'] = test
                yield tests[test]

def save_interview(input, templateout, params, model, results):
    [stage, interview_name, languages, template, *stuff] = Path(input).stem.split('_')
    templateout_name = Path(templateout).stem
    params_name = Path(params).stem
    model_name = model.replace('/','-').replace('_','-')
    ts = str(int(time.time()))

    output_filename = str(Path(input).parent)+'/'+'_'.join(['interview', interview_name, languages, template, templateout_name, params_name, model_name, ts])+'.ndjson'
    with open(output_filename, 'w') as f:
        f.write('\n'.join([json.dumps(result, default=vars) for result in results]))
    print('Saved results to', output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interview preparation')
    parser.add_argument('--language', type=str, default='python,javascript', help='languages to prepare, comma seperated')
    parser.add_argument('--interview', type=str, default='junior-v2', help='interview to prepare')
    parser.add_argument('--template', type=str, required=True, help='prompt template file')
    args = parser.parse_args()

    template = Template(open(args.template).read())
    template_name = Path(args.template).stem

    output_filename = f"results/prepare_{args.interview}_{args.language.replace(',', '-')}_{template_name}.ndjson"
    outputs = []
    for test in load_questions(interview=args.interview):
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