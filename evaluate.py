#!/usr/bin/env python3
import argparse
import json
import glob
import logging
import os

from threading import Thread
from queue import Queue

from prepare import load_questions
from extract import extract_code
from sbox.sandbox import FunctionSandbox, build_sandbox, start_sandbox, stop_sandbox

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
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )

    return logging.getLogger("evaluator")

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
        
    # Configure logging for threading
    # No need for multiprocessing-specific logging with ThreadPoolExecutor

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
        
    # Build sandbox images once at the beginning
    languages = ['python', 'javascript']
    for language in languages:
        build_sandbox(language, logger)
    
    # Filter files that need processing, place into a queue.
    file_queue = Queue()
    for input_file in input_files:
        output_filename = input_file.replace('interview','eval')
        
        # Skip if output file already exists and --rerun wasn't specified
        if os.path.exists(output_filename) and not args.rerun:
            logger.info(f'Skipped {input_file}, output file {output_filename} already exists.')
            continue
            
        # Also check if the input file itself has already been processed
        try:
            with open(input_file) as f:
                first_line = f.readline()
                data = json.loads(first_line)
                if 'code' not in data or args.rerun:
                    file_queue.put((input_file, output_filename))
                else:
                    logger.info(f'Skipped {input_file}, already processed.')
        except Exception as e:
            logger.error(f"Error reading {input_file}: {e}")
   
    if file_queue.qsize() == 0:
        logger.info("No files need processing. Use --rerun to force reprocessing.")
        exit(0)
    
    logger.info(f"Found {file_queue.qsize()} files that need processing")
       
    # Define a worker function that processes files from the queue
    def worker_process(worker_id):
        worker_logger = logging.getLogger(f"worker-{worker_id}")
        worker_logger.info(f"Worker {worker_id} started")
        
        # Start sandbox containers for this worker
        for language in languages:
            start_sandbox(language, worker_id, worker_logger)
        
        worker_total = { 'javascript': 0, 'python': 0 }
        worker_passed = { 'javascript': 0, 'python': 0 }
        
        try:
            while not file_queue.empty():
                try:
                    input_file, output_filename = file_queue.get(block=False)
                    worker_logger.info(f"Processing file: {input_file}")
                    
                    results = []
                    file_total = { 'javascript': 0, 'python': 0 }
                    file_passed = { 'javascript': 0, 'python': 0 }
                
                    answers = [json.loads(line) for line in open(input_file)]
                
                    for test in answers:
                        if args.test and test['name'] != args.test:
                            worker_logger.info(f"{test['name']} - Skipped due to command line filter")
                            continue

                        code = extract_code(test['answer'], stop_at_prefix)
                    
                        if code:
                            worker_logger.debug(f"{test['name']} - {test['language']} - started (instance {worker_id})")
                        else:
                            worker_logger.warning(f"{test['name']} - {test['language']} - extract_code failed")
                            worker_logger.debug(f"Answer: {test['answer']}")

                        total, passed, checks, status = evaluation(interview[test['name']], test['language'], code, worker_id, worker_logger)

                        file_total[test['language']] += total
                        file_passed[test['language']] += passed
                        worker_total[test['language']] += total
                        worker_passed[test['language']] += passed

                        row = test.copy()
                        row['code'] = code
                        row['checks'] = checks
                        row['status'] = status
                        row['passed'] = passed
                        row['total'] = total
                        results.append(row)

                        worker_logger.info(f"{row['name']} - {test['language']} - {row['status']}")

                    if not args.test and results:
                        with open(output_filename,'w') as f:
                            f.write('\n'.join([json.dumps(r) for r in results]))
                        worker_logger.info(f"File: {input_file}")
                        worker_logger.info(f"Python Passed {file_passed['python']} of {file_total['python']}")
                        worker_logger.info(f"JavaScript Passed {file_passed['javascript']} of {file_total['javascript']}")
                        worker_logger.info(f"Evaluation results written to {output_filename}")
                    
                    file_queue.task_done()
                except Queue.Empty:
                    break
                except Exception as e:
                    worker_logger.error(f"Error processing file {input_file}: {e}")
                    file_queue.task_done()
        finally:
            # Stop sandbox instances for this worker
            worker_logger.info(f"Worker {worker_id} finished, stopping sandbox instances")
            for language in languages:
                stop_sandbox(language, worker_id, worker_logger)
        
        return worker_total, worker_passed
    
    # Start worker threads   
    workers = []
    results = []    
    for i in range(min(args.parallel, file_queue.qsize())):
        thread = Thread(target=lambda i=i: results.append(worker_process(i)))
        thread.start()
        workers.append(thread)
    
    # Wait for all workers to finish
    for thread in workers:
        thread.join()
    
    # Aggregate results
    for worker_total, worker_passed in results:
        for language in worker_total:
            all_total[language] += worker_total[language]
            all_passed[language] += worker_passed[language]
    
    if len(input_files) > 1:
        logger.info("Overall Summary:")
        logger.info(f"Python Passed {all_passed['python']} of {all_total['python']}")
        logger.info(f"JavaScript Passed {all_passed['javascript']} of {all_total['javascript']}")
