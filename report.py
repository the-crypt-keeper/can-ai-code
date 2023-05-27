#!/usr/bin/env python3
import os
import json
import sys

def calculate_totals(folder_path):
    python_file_path = os.path.join(folder_path, 'eval-python.json')
    javascript_file_path = os.path.join(folder_path, 'eval-javascript.json')

    if not os.path.isfile(python_file_path):
        print(f"File 'eval-python.json' not found in {folder_path}")
        return

    if not os.path.isfile(javascript_file_path):
        print(f"File 'eval-javascript.json' not found in {folder_path}")
        return

    python_data = load_json_file(python_file_path)
    javascript_data = load_json_file(javascript_file_path)

    python_total_sum, python_passed_sum = calculate_sum(python_data)
    javascript_total_sum, javascript_passed_sum = calculate_sum(javascript_data)

    print(f"Python Total Sum: {python_total_sum}")
    print(f"Python Passed Sum: {python_passed_sum}")
    print(f"JavaScript Total Sum: {javascript_total_sum}")
    print(f"JavaScript Passed Sum: {javascript_passed_sum}")

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError:
        print(f"Failed to parse {file_path} as JSON.")

    return []

def calculate_sum(data):
    total_sum = 0
    passed_sum = 0

    for item in data:
        if 'total' in item and 'passed' in item:
            total_sum += item['total']
            passed_sum += item['passed']

    return total_sum, passed_sum

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the folder path as an argument.")
    else:
        folder_path = sys.argv[1]
        calculate_totals(folder_path)
