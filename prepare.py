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

def prepare_interview(interview, languages, message_template, template_name, tokenizer):
    output_filename = f"results/prepare_{interview}_{languages.replace(',', '-')}_{template_name}.ndjson"
    outputs = []
    for test in load_questions(interview=interview):
        for language in languages.split(','):
            messages = []
            for msg in message_template:
                content = msg['content'].render({'language': language, **test})
                messages.append({'role': msg['role'], 'content': content})
                
            if tokenizer:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = '\n'.join([msg['content'] for msg in messages])
                
            output = test.copy()
            del output['Checks']
            output['language'] = language
            output['prompt'] = prompt
            outputs.append(output)
            
    return output_filename,outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interview preparation')
    parser.add_argument('--language', type=str, default='python,javascript', help='languages to prepare, comma seperated')
    parser.add_argument('--interview', type=str, default='junior-v2,senior', help='interviews to prepare')
    parser.add_argument('--template', type=str, help='prompt template file')
    parser.add_argument('--chat',type=str, help='outer chat prompt huggingface model name')
    args = parser.parse_args()
    
    if args.chat and not args.template: args.template = 'prompts/chat-simple.txt'
    assert(args.template)
    
    template_text = open(args.template).read()
    if 'json' in args.template:
        message_template = json.loads(template_text)
    else:
        message_template = [{'role': 'user', 'content': template_text}]
    for msg in message_template: msg['content'] = Template(msg['content'])
    
    if args.chat:
        from transformers import AutoTokenizer
        template_name = Path(args.template).stem+'-'+args.chat.replace('/','-').replace('_','-')
        tokenizer = AutoTokenizer.from_pretrained(args.chat, trust_remote_code=True)
    else:
        template_name = Path(args.template).stem
        tokenizer = None
   
    for interview in args.interview.split(','):
        output_filename, outputs = prepare_interview(interview, args.language, message_template, template_name, tokenizer)
        with open(output_filename, 'w') as file:
            file.write('\n'.join([json.dumps(output) for output in outputs]))
            print(f"Expanded {len(outputs)} {template_name} prompts to {output_filename}")