#!/usr/bin/env python3
from prepare import load_questions
from sbox.sandbox import FunctionSandbox
import argparse
import json
import re

def extract_code(answer):
    # Fallback if the model forgot to use block quotes or used a single quote instead.
    simple_answer = answer.replace('`','').strip()
    if simple_answer[0:3] == 'def' or simple_answer[0:8] == 'function':
        return simple_answer

    # Look for start tokens   
    match = re.search(r'```(\w*)', answer)
    start_token = match.group(0) if match else None
    start_index = match.start() if match else -1

    # If we didn't find a start token, return None
    if start_index == -1:
        return None

    # Find the index of the end token, starting from the end of the start token.
    # if not found, assume we're taking the whole thing.
    end_token = "```"
    end_index = answer.find(end_token, start_index + len(start_token))
    if end_index == -1:
        end_index = len(answer)

    # Extract the text between the tokens
    code_text = answer[start_index + len(start_token):end_index].strip()

    return code_text

def evaluation(test, language, code):
    total = 0
    passed = 0
    checks = []

    if code:
        f = FunctionSandbox(code, language)

        for check_name in test['Checks'].keys():
            check = test['Checks'][check_name]
            if check.get('assert'):
                total += 1
                test_value = None
                try:
                    test_value = eval(check['assert'])
                except Exception as e:
                    test_value = str(e)

                check['got'] = test_value

                if (test_value == check['eq']):
                    print('   ',check_name, "passed")
                    passed += 1
                    check['status'] = 1
                else:
                    check['status'] = 0
                    print('   ',check_name, "failed", check['assert'], 'got', test_value, '!=', check['eq'])
            checks.append(check)
    else:
        print(test['name'], "No code found!")
        total = len(test['Checks'].keys())

    return total,passed,checks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interview evaluator')
    parser.add_argument('--interview', type=str, default='junior-dev', help='interview to evaluate')
    parser.add_argument('--input', type=str, required=True, help='path to interview*.ndjson')
    parser.add_argument('--test', type=str, help='(optional) specific test to evaluate')
    parser.add_argument('--noextract', action='store_true', help='(optional) skip code extraction')
    args = parser.parse_args()

    all_total = { 'javascript': 0, 'python': 0 }
    all_passed = { 'javascript': 0, 'python': 0 }
    results = []

    interview = {}
    for test in load_questions(args.interview):
        interview[test['name']] = test

    answers = [json.loads(line) for line in open(args.input)]
    for test in answers:

        if args.test and test['name'] != args.test:
            print(test_name, 'Skipped due to command line filter')
            continue

        code = extract_code(test['answer']) if not args.noextract else test['answer']
        
        if code:
            print(test['name'], test['language'], 'started')
        else:
            print(test['name'], test['language'], 'extract_code failed')
            print(test['answer'])

        total, passed, checks = evaluation(interview[test['name']], test['language'], code)

        all_total[test['language']] += total
        all_passed[test['language']] += passed

        row = test.copy()
        row['code'] = code
        row['checks'] = checks
        row['status'] = 'NOCODE' if (not code) else 'PASS' if passed == total else 'FAIL'
        row['passed'] = passed
        row['total'] = total
        results.append(row)

        print(row['name'], test['language'], row['status'])
        print()

    FunctionSandbox.stopall()

    if not args.test:
        output_filename = args.input.replace('interview','eval')
        with open(output_filename,'w') as f:
            f.write('\n'.join([json.dumps(r) for r in results]))
        print('Python Passed',all_passed['python'],'of',all_total['python'])
        print('JavaScript Passed',all_passed['javascript'],'of',all_total['javascript'])
        print('Evaluation results written to',output_filename)