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
                prompt = messages

            output = test.copy()
            del output['Checks']
            output['language'] = language
            output['prompt'] = prompt
            outputs.append(output)
            
    return output_filename,outputs

def cli_to_interviews(input, interview, tokenizer, prompt = 'prompts/chat.json'):
    interviews = []
    if input != "" and input is not None:
        for input_file in input.split(','):
            interview = [json.loads(line) for line in open(input_file)]
            interviews.append( (input_file, interview) )
            print(f"Loaded {len(interview)} questions from {input_file}.")
    elif interview != "":
        for interview_name in interview.split(','):
            language = "python,javascript"
            template_name = prompt.split('/')[-1] if '/' in prompt else prompt
            template_name = template_name.replace('.json','')
            with open(prompt) as f:
                message_template = json.load(f)
            for msg in message_template:
                msg['content'] = Template(msg['content'])
            output_filename, interview = prepare_interview(interview_name, language, message_template, template_name, tokenizer)
            interviews.append( (output_filename, interview) )
            print(f"Expanded {len(interview)} questions from {interview_name}.")
    else:
        raise Exception("Please provide either --input or --interview")
    
    return interviews
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interview preparation')
    parser.add_argument('--language', type=str, default='python,javascript', help='languages to prepare, comma seperated')
    parser.add_argument('--interview', type=str, default='junior-v2,senior', help='interviews to prepare')
    parser.add_argument('--prompt', type=str, help='prompt template file')
    parser.add_argument('--chat',type=str, help='outer chat prompt huggingface model name')
    args = parser.parse_args()
    
    if args.chat and not args.prompt: args.prompt = 'prompts/chat.json'
    assert(args.prompt)
    
    template_text = open(args.prompt).read()
    if 'json' in args.prompt:
        message_template = json.loads(template_text)
    else:
        message_template = [{'role': 'user', 'content': template_text}]
    for msg in message_template: msg['content'] = Template(msg['content'])
    
    if args.chat:
        from transformers import AutoTokenizer
        template_name = Path(args.prompt).stem+'-'+args.chat.replace('/','-').replace('_','-')
        tokenizer = AutoTokenizer.from_pretrained(args.chat, trust_remote_code=True)
    else:
        template_name = Path(args.prompt).stem
        tokenizer = None
   
    for interview in args.interview.split(','):
        output_filename, outputs = prepare_interview(interview, args.language, message_template, template_name, tokenizer)
        with open(output_filename, 'w') as file:
            file.write('\n'.join([json.dumps(output) for output in outputs]))
            print(f"Expanded {len(outputs)} {template_name} prompts to {output_filename}")