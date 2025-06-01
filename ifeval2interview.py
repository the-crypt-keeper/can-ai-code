#!/usr/bin/env python3
"""
Convert IFEval JSONL format to interview format.

Takes a JSONL file with 'key' and 'prompt' fields and converts it to a JSON object
where each key maps to an object with the prompt.
"""

import json
import argparse
import sys


def convert_ifeval_to_interview(input_path, output_path=None):
    """
    Convert IFEval JSONL format to interview format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSON file (optional, defaults to stdout)
    """
    result = {}
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if 'key' not in data or 'prompt' not in data:
                        print(f"Warning: Line {line_num} missing 'key' or 'prompt' field", file=sys.stderr)
                        continue
                        
                    key = data['key']
                    prompt = data['prompt']
                    
                    result[key] = {
                        'prompt': prompt
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}", file=sys.stderr)
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output the result
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"Converted {len(result)} entries to {output_path}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_json)


def main():
    parser = argparse.ArgumentParser(description='Convert IFEval JSONL to interview format')
    parser.add_argument('input', help='Input JSONL file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: stdout)')
    
    args = parser.parse_args()
    
    convert_ifeval_to_interview(args.input, args.output)


if __name__ == '__main__':
    main()
