#!/usr/bin/env python3
from interview import load_questions
from sandbox import FunctionSandbox
import argparse

parser = argparse.ArgumentParser(description='Interview evaluator')
parser.add_argument('--language', type=str, required=True, help='language to use')
parser.add_argument('--answers', type=str, required=True, help='path to model answers')
args = parser.parse_args()

def extract_code(answer):
    start_token = "```python"
    end_token = "```"

    # Find the index of the start token
    start_index = answer.find(start_token)
    if start_index == -1:
        return None

    # Find the index of the end token, starting from the end of the start token
    end_index = answer.find(end_token, start_index + len(start_token))
    if end_index == -1:
        return None

    # Extract the text between the tokens
    code_text = answer[start_index + len(start_token):end_index].strip()

    return code_text

for test in load_questions():
    answer = None
    test_name = test['name'] + '-' + args.language

    try:
        with open(args.answers+test_name+'.txt','r') as f:
            answer = f.read()
    except Exception as e:
        print(e)
        continue

    code = extract_code(answer)
    if code:
        f = FunctionSandbox(code)
        total = 0
        passed = 0
        print(test_name+' started')
        #print('---\n'+code+'\n---')
        for check_name in test['Checks'].keys():
            check = test['Checks'][check_name]
            if check.get('assert'):
                total += 1
                test_value = None
                try:
                    test_value = eval(check['assert'])
                except Exception as e:
                    test_value = str(e)

                if (test_value == check['eq']):
                    print('   ',check_name, "passed")
                    passed += 1
                else:
                    print('   ',check_name, "failed", check['assert'], 'got', test_value, '!=', check['eq'])
        print(test_name,'passed',passed,'of',total)
        print()
    else:
        print("No code found")