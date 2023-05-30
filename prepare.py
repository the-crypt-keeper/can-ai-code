#!/usr/bin/env python3
import glob
import yaml
import argparse
import sys
from jinja2 import Template
import pandas as pd

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
    parser.add_argument('--questions', type=str, help='output  to .csv file (default console)')
    args = parser.parse_args() 

    output = []
    for test in load_questions():
        for language in args.language.split(','):
            test_name = test['name'] + '-' + language

            if isinstance(test['Request'], str):
                test_prompt = Template(test['Request']).render(language=language)
            else:
                test_prompt = test['Request'].get(language)
            
            if test_prompt is None:
                print('WARNING: Skipped ',test['name'],'because no prompt could be found for',language)
                continue
            
            output.append({'name': test_name, 'language': language, 'prompt': test_prompt})

    df = pd.DataFrame.from_records(output)
    df.to_csv(args.questions if args.questions else sys.stdout, index=False)
