#!/usr/bin/env python3
from prepare import load_questions
from sbox.sandbox import FunctionSandbox
import argparse
import json
import os
from extract import extract_code

def evaluation(test, language, code):
    total = len(test['Checks'].keys())
    passed = 0
    checks = []

    if not code:
        print(test['name'], "No code found!")
        return total,passed,checks,"NO_CODE"
    
    f = FunctionSandbox(code, language)
    if f.functions['name'] == '':
        print(test['name'], "No function found!")
        return total,passed,checks,"NO_FUNCTION"

    for check_name in test['Checks'].keys():
        check = test['Checks'][check_name].copy()
        if not check.get('assert'):
            raise Exception(f'check {check_name} missing assert')

        test_value = None
        try:
            test_value = eval(check['assert'])
        except Exception as e:
            test_value = str(e)

        check['got'] = test_value

        test_result = (test_value in check['eq-any']) if ('eq-any' in check) else (test_value == check['eq'])           
        if (test_result):
            passed += 1
            check['status'] = 1
            print('   ',check_name, "passed", test_value,
                    'inside' if 'eq-any' in check else '==', check.get('eq', check.get('eq-any')))
        else:
            check['status'] = 0
            print('   ',check_name, "failed", check['assert'], 'got', test_value,
                    'not inside' if 'eq-any' in check else '!=', check.get('eq', check.get('eq-any')))
            
        checks.append(check)

    return total,passed,checks,"PASS" if (total==passed) else "FAIL"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interview evaluator')
    parser.add_argument('--interview', type=str, default='junior-v2', help='interview to evaluate')
    parser.add_argument('--input', type=str, required=True, help='path to interview*.ndjson')
    parser.add_argument('--test', type=str, help='(optional) specific test to evaluate')
    parser.add_argument('--stopcomment', action='store_true', help='(optional) stop code extraction at first comment')
    args = parser.parse_args()

    all_total = { 'javascript': 0, 'python': 0 }
    all_passed = { 'javascript': 0, 'python': 0 }
    results = []
    stop_at_prefix = ['//','#'] if args.stopcomment else []

    interview = {}
    for test in load_questions(args.interview):
        interview[test['name']] = test

    answers = [json.loads(line) for line in open(args.input)]
    for test in answers:

        if args.test and test['name'] != args.test:
            print(test['name'], 'Skipped due to command line filter')
            continue

        code = extract_code(test['answer'], stop_at_prefix)
        
        if code:
            print(test['name'], test['language'], 'started')
        else:
            print(test['name'], test['language'], 'extract_code failed')
            print(test['answer'])

        total, passed, checks, status = evaluation(interview[test['name']], test['language'], code)

        all_total[test['language']] += total
        all_passed[test['language']] += passed

        row = test.copy()
        row['code'] = code
        row['checks'] = checks
        row['status'] = status
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