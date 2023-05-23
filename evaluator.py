#!/usr/bin/env python3
from interview import load_questions
from sandbox import FunctionSandbox

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

ANSWER_DIR = "result-0.0/"

for test in load_questions():
    answer = None
    with open(ANSWER_DIR+test['name']+'.txt','r') as f:
        answer = f.read()

    print(test)
    code = extract_code(answer)
    if code:
        f = FunctionSandbox(code)
        for check_name in test['Checks'].keys():
            check = test['Checks'][check_name]
            if check.get('assert'):
                passed = False
                test_value = None

                try:
                    test_value = eval(check['assert'])
                except Exception as e:
                    test_value = str(e)
                
                if (test_value == check['eq']):
                    passed = True

                if passed:
                    print(check_name, "passed")
                else:
                    print(check_name, "failed", check['assert'], 'got', test_value, '!=', check['eq'])
    else:
        print("No code found")