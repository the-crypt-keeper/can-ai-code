#!/usr/bin/env python3
import glob
import yaml

def load_questions():
    for file_path in glob.glob('questions/*.yaml'):
        with open(file_path, 'r') as file:
            tests = yaml.safe_load(file)
            for test in tests.keys():
                if test[0] == '.':
                    continue
                tests[test]['name'] = test
                yield tests[test]

if __name__ == "__main__":
    print("name,prompt")
    for test in load_questions():
        print(test['name']+",\""+test['Request']+"\"")