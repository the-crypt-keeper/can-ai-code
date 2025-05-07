#!/usr/bin/env python3
from prepare import load_questions
from sbox.sandbox import FunctionSandbox
import argparse
import json
import os
import glob
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from extract import extract_code
from termcolor import colored

def evaluation(test, language, code, instance_id=0, logger=None):
    if logger is None:
        logger = logging.getLogger(f"eval-{instance_id}")
        
    total = sum([check.get('weight',1) for _, check in test['Checks'].items()])
    passed = 0
    checks = []

    if not code:
        logger.warning(f"{test['name']} - No code found!")
        return total,passed,checks,"NO_CODE"
    
    f = FunctionSandbox(code, language, instance_id, logger)
    if f.functions['name'] == '':
        logger.warning(f"{test['name']} - No function found!")
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
                    if errors_pre < errors: 
                        errors = errors_pre
                        ratio = ratio_pre
                else:
                    # more test values then check values, pad check
                    errors, ratio = compare_lists(check_val+[None], test_value)
                    errors_pre, ratio_pre = compare_lists([None]+check_val, test_value)
                    if errors_pre < errors: 
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

        # print(colored(f'  [{weight}/{max_weight}] {check_result:4} {check_name:20} {test_value} {check_op} {check_val}', 'red' if not test_result else 'green'))
        checks.append(check)

    return total,passed,checks,"PASS" if (total==passed) else "FAIL"

def setup_logging(level=logging.INFO):
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("evaluator")

def start_sandboxes(num_instances, languages=['python', 'javascript']):
    """Start all required sandbox instances at once - deprecated, now each thread manages its own"""
    logger = logging.getLogger("sandbox-starter")
    logger.warning("This function is deprecated - each thread now manages its own sandbox instances")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interview evaluator')
    parser.add_argument('--interview', type=str, default='junior-v2', help='interview to evaluate')
    parser.add_argument('--input', type=str, help='path to interview*.ndjson')
    parser.add_argument('--glob', type=str, help='glob pattern for multiple input files')
    parser.add_argument('--test', type=str, help='(optional) specific test to evaluate')
    parser.add_argument('--stopcomment', action='store_true', help='(optional) stop code extraction at first comment')
    parser.add_argument('--rerun', action='store_true', help='(optional) rerun evaluation on already processed files')
    parser.add_argument('--parallel', type=int, default=4, help=f'number of parallel processes to use (default: 4)')
    parser.add_argument('--verbose', action='store_true', help='enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    if not args.input and not args.glob:
        parser.error("Either --input or --glob must be provided")
        
    # Configure multiprocessing logging
    multiprocessing.log_to_stderr(log_level)

    all_total = { 'javascript': 0, 'python': 0 }
    all_passed = { 'javascript': 0, 'python': 0 }
    stop_at_prefix = ['//','#'] if args.stopcomment else []

    interview = {}
    for test in load_questions(args.interview):
        interview[test['name']] = test

    # Get list of input files
    input_files = []
    if args.input:
        input_files = [args.input]
    elif args.glob:
        input_files = glob.glob(args.glob)
        if not input_files:
            logger.error(f"No files found matching pattern: {args.glob}")
            exit(1)
        logger.info(f"Processing {len(input_files)} files matching pattern: {args.glob}")
        
    # No longer starting all sandbox instances at the beginning
    # Each thread will start its own sandbox

    def process_file_batch(batch_data):
        thread_id, file_batch, interview_data, test_filter, stop_prefixes, rerun = batch_data
    
        # Setup thread-specific logger
        thread_logger = logging.getLogger(f"thread-{thread_id}")
        thread_logger.info(f"Thread {thread_id} processing {len(file_batch)} files")
    
        # Start sandbox instances for this thread
        instance_id = thread_id
        languages = ['python', 'javascript']
        for language in languages:
            FunctionSandbox.start_sandbox(language, instance_id, thread_logger)
    
        batch_results = []
        batch_total = { 'javascript': 0, 'python': 0 }
        batch_passed = { 'javascript': 0, 'python': 0 }
    
        try:
            # Process each file in the batch
            for input_file in file_batch:
                thread_logger.info(f"Processing file: {input_file}")
            
                results = []
                file_total = { 'javascript': 0, 'python': 0 }
                file_passed = { 'javascript': 0, 'python': 0 }
            
                answers = [json.loads(line) for line in open(input_file)]
            
                # Check if file has already been processed
                if 'code' in answers[0] and not rerun:
                    thread_logger.info(f"File {input_file} has already been processed. Use --rerun to process again.")
                    continue
            
                for test in answers:
                    if test_filter and test['name'] != test_filter:
                        thread_logger.info(f"{test['name']} - Skipped due to command line filter")
                        continue

                    code = extract_code(test['answer'], stop_prefixes)
                
                    if code:
                        thread_logger.debug(f"{test['name']} - {test['language']} - started (instance {instance_id})")
                    else:
                        thread_logger.warning(f"{test['name']} - {test['language']} - extract_code failed")
                        thread_logger.debug(f"Answer: {test['answer']}")

                    total, passed, checks, status = evaluation(interview_data[test['name']], test['language'], code, instance_id, thread_logger)

                    file_total[test['language']] += total
                    file_passed[test['language']] += passed
                    batch_total[test['language']] += total
                    batch_passed[test['language']] += passed

                    row = test.copy()
                    row['code'] = code
                    row['checks'] = checks
                    row['status'] = status
                    row['passed'] = passed
                    row['total'] = total
                    results.append(row)

                    thread_logger.info(f"{row['name']} - {test['language']} - {row['status']}")

                if not test_filter and results:
                    output_filename = input_file.replace('interview','eval')
                    with open(output_filename,'w') as f:
                        f.write('\n'.join([json.dumps(r) for r in results]))
                    thread_logger.info(f"File: {input_file}")
                    thread_logger.info(f"Python Passed {file_passed['python']} of {file_total['python']}")
                    thread_logger.info(f"JavaScript Passed {file_passed['javascript']} of {file_total['javascript']}")
                    thread_logger.info(f"Evaluation results written to {output_filename}")
                
                batch_results.append((results, file_total, file_passed))
    
        finally:
            # Stop sandbox instances for this thread
            thread_logger.info(f"Thread {thread_id} finished, stopping sandbox instances")
            for language in languages:
                FunctionSandbox.stop_sandbox(language, instance_id, thread_logger)
    
        return batch_results, batch_total, batch_passed

    # Filter files that need processing
    files_to_process = []
    for input_file in input_files:
        try:
            with open(input_file) as f:
                first_line = f.readline()
                data = json.loads(first_line)
                if 'code' not in data or args.rerun:
                    files_to_process.append(input_file)
        except Exception as e:
            logger.error(f"Error checking file {input_file}: {e}")
            files_to_process.append(input_file)
    
    if not files_to_process:
        logger.info("No files need processing. Use --rerun to force reprocessing.")
        return
    
    logger.info(f"Found {len(files_to_process)} files that need processing")
    
    # Split files into batches for parallel processing
    num_batches = min(args.parallel, len(files_to_process))
    file_batches = [[] for _ in range(num_batches)]
    
    for i, file_path in enumerate(files_to_process):
        batch_idx = i % num_batches
        file_batches[batch_idx].append(file_path)
    
    # Prepare batch data for parallel processing
    batch_data_list = []
    for batch_idx, file_batch in enumerate(file_batches):
        if file_batch:  # Only process non-empty batches
            batch_data_list.append((batch_idx, file_batch, interview, args.test, stop_at_prefix, args.rerun))
    
    logger.info(f"Splitting work into {len(batch_data_list)} batches")
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=len(batch_data_list)) as executor:
        batch_results = list(executor.map(process_file_batch, batch_data_list))
    
    # Aggregate results
    for batch_file_results, batch_total, batch_passed in batch_results:
        for language in batch_total:
            all_total[language] += batch_total[language]
            all_passed[language] += batch_passed[language]
    
    if len(input_files) > 1:
        logger.info("Overall Summary:")
        logger.info(f"Python Passed {all_passed['python']} of {all_total['python']}")
        logger.info(f"JavaScript Passed {all_passed['javascript']} of {all_total['javascript']}")
