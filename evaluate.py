#!/usr/bin/env python3
from prepare import load_questions
from sbox.sandbox import FunctionSandbox
import argparse
import json
import os
from extract import extract_code
from termcolor import colored

def evaluation(test, language, code):
    total = sum([check.get('weight',1) for _, check in test['Checks'].items()])
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
        check_val = check.get('eq', check.get('eq-any'))
                
        if check.get('eq-any'):
            test_result = test_value in check['eq-any']
            ratio = 0 if not test_result else 1
        elif isinstance(check_val, str) or isinstance(check_val, int):
            test_result = test_value == check['eq']
            ratio = 0 if not test_result else 1
        elif isinstance(check_val, dict):
            if not isinstance(test_value, dict):
                errors, ratio = 1, 0
            else:
                errors, good = 0,0
                for key, value in check_val.items():
                    if test_value.get(key) != value: 
                        errors += 1
                    else:
                        good += 1
                ratio = good/(good+errors)
            test_result = (errors == 0)
        elif isinstance(check_val, list):

            def compare_lists(l1, l2):
                bad, good = 0, 0
                for idx in range(max(len(l1),len(l2))):
                    item1 = l1[idx] if idx<len(l1) else None
                    item2 = l2[idx] if idx<len(l2) else None
                    if item1 != item2:
                        bad += 1
                    else:
                        good += 1
                return bad, good/(bad+good)
            
            # lists are same size
            if not isinstance(test_value, list):
                errors, ratio = 1, 0
            elif len(check_val) == len(test_value):
                errors, ratio = compare_lists(check_val, test_value)
            else:
                # try to gracefully handle off-by-ones without failing the whole list
                if len(check_val) > len(test_value):
                    # more check values then test values, pad test
                    errors, ratio = compare_lists(check_val, test_value+[None])
                    errors_pre, ratio_pre = compare_lists(check_val, [None]+test_value)
                    if errors_pre > errors: 
                        errors = errors_pre
                        ratio = ratio_pre
                else:
                    # more test values then check values, pad check
                    errors, ratio = compare_lists(check_val+[None], test_value)
                    errors_pre, ratio_pre = compare_lists([None]+check_val, test_value)
                    if errors_pre > errors: 
                        errors = errors_pre
                        ratio = ratio_pre
       
            test_result = (errors == 0)
        
        max_weight = check.get('weight', 1)
        weight = int(max_weight*ratio)
        passed += weight
        if (test_result):
            check['status'] = weight
            check_result = 'pass'
            check_op = 'inside' if 'eq-any' in check else '=='            
        else:
            check['status'] = weight
            check_result = 'FAIL'
            check_op = 'not inside' if 'eq-any' in check else '!='

        print(colored(f'  [{weight}/{max_weight}] {check_result:4} {check_name:20} {test_value} {check_op} {check_val}', 'red' if not test_result else 'green'))
        checks.append(check)

    return total,passed,checks,"PASS" if (total==passed) else "FAIL"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interview evaluator')
    parser.add_argument('--interview', type=str, default='junior-v2', help='interview to evaluate')
    parser.add_argument('--input', type=str, required=True, help='path to interview*.ndjson')
    parser.add_argument('--test', type=str, help='(optional) specific test to evaluate')
    parser.add_argument('--stopcomment', action='store_true', help='(optional) stop code extraction at first comment')
    parser.add_argument('--persist_sandbox', action='store_true', help='(optional) leave sandbox running')
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

    if not args.persist_sandbox:
        FunctionSandbox.stopall()

    if not args.test:
        output_filename = args.input.replace('interview','eval')
        with open(output_filename,'w') as f:
            f.write('\n'.join([json.dumps(r) for r in results]))
        print('Python Passed',all_passed['python'],'of',all_total['python'])
        print('JavaScript Passed',all_passed['javascript'],'of',all_total['javascript'])
        print('Evaluation results written to',output_filename)