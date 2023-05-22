#!/usr/bin/env python3
import glob
import yaml
from jinja2 import Template

def load_questions():
    for file_path in glob.glob('questions/*.yaml'):
        with open(file_path, 'r') as file:
            tests = yaml.safe_load(file)
            for test in tests.keys():
                if test[0] == '.':
                    continue
                tests[test]['name'] = test
                yield tests[test]

def generate_questions():
    languages = ['python', 'javascript', 'typescript']
    
    for question in load_questions():
        for language in languages:
            vars = {}
            for var in question.get('Variables',{}).keys():
                vars[var] = question['Variables'][var][language]
            rendered_template = Template(question['Request']).render(language=language, **vars)
            yield question['name'], language, rendered_template

print("name,prompt")
for test in load_questions():
    print(test['name']+",\""+test['Request']+"\"")