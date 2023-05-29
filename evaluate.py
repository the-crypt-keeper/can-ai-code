#!/usr/bin/env python3
from prepare import load_questions
from sbox.sandbox import FunctionSandbox
import argparse
import json
import re

def extract_code(answer):
    # Fallback if the model forgot to use block quotes
    if answer.strip()[0:3] == 'def' or answer.strip()[0:8] == 'function':
        return answer

    # Look for start tokens   
    match = re.search(r'```(\w*)', answer)
    start_token = match.group(0) if match else None
    start_index = match.start() if match else -1

    # If we didn't find a start token, return None
    if start_index == -1:
        return None

    # Find the index of the end token, starting from the end of the start token
    end_token = "```"
    end_index = answer.find(end_token, start_index + len(start_token))
    if end_index == -1:
        return None

    # Extract the text between the tokens
    code_text = answer[start_index + len(start_token):end_index].strip()

    return code_text

def evaluation(test, language, code):
    total = 0
    passed = 0
    checks = []

    if code:
        f = FunctionSandbox(code, language)
        print(test_name,'started')
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
        print(test_name, "No code found")
        passed = -1
    return total,passed,checks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interview evaluator')
    parser.add_argument('--interview', type=str, default='junior-dev', help='interview to evaluate')
    parser.add_argument('--language', type=str, required=True, help='language to evaluate')
    parser.add_argument('--answers', type=str, required=True, help='path to model answers')
    parser.add_argument('--test', type=str, help='(optional) specific test to evaluate')
    parser.add_argument('--noextract', action='store_true', help='(optional) skip code extraction')
    args = parser.parse_args()

    all_total = 0
    all_passed = 0
    results = []

    for test in load_questions(args.interview):
        answer = None
        test_name = test['name'] + '-' + args.language

        if args.test and test_name  != args.test:
            print(test_name, 'Skipped due to command line filter')
            continue

        try:
            with open(args.answers+test_name+'.txt','r') as f:
                answer = f.read()
        except Exception as e:
            print(test_name,'Skipped due to error', e)
            print()
            row = { 'test': test_name, 'language': args.language, 'status': 'ERROR', 'error': str(e) }
            results.append(row)
            continue

        code = extract_code(answer) if not args.noextract else answer
        total, passed, checks = evaluation(test, args.language, code)

        all_total += total
        all_passed += passed

        row = { 'test': test_name, 'language': args.language, 'checks': checks, 'status': 'PASS' if passed==total else 'NOCODE' if passed == -1 else 'FAIL', 'error': None, 'answer': answer, 'code': code,  'passed': passed, 'total': total }
        results.append(row)
        print(row['test'], row['status'])
        print()

    if not args.test:
        outfn = f"{args.answers}eval-{args.language}.json"
        with open(outfn,'w') as f:
            json.dump(results, f, indent=2)
        print('Passed',all_passed,'of',all_total,'results written to',outfn)    