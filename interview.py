#!/usr/bin/env python3
import glob
import yaml
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interview questions')
    parser.add_argument('--language', required=True, help='language to use')
    args = parser.parse_args()

    print("name,prompt")
    for test in load_questions():
        test_name = test['name'] + '-' + args.language

        if isinstance(test['Request'], str):
            test_prompt = Template(test['Request']).render(language=args.language)
        else:
            test_prompt = test['Request'][args.language]
        
        print(test_name + ',\"' + test_prompt + '\"')