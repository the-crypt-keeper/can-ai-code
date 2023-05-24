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
    try:
        with open(ANSWER_DIR+test['name']+'.txt','r') as f:
            answer = f.read()
    except Exception as e:
        print(e)
        continue

    code = extract_code(answer)
    if code:
        f = FunctionSandbox(code)
        total = 0
        passed = 0
        print('\n\n'+test['name']+' started')
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
        print(test['name'],'passed',passed,'of',total)
    else:
        print("No code found")