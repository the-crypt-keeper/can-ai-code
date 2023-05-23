import re
from typing import Any

def extract_function_info(input_string):
    function_regex = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\):"
    matches = re.findall(function_regex, input_string, re.MULTILINE)

    functions = []
    for match in matches:
        function_name = match[0]
        arguments = match[1].split(',')

        # Extract argument names by removing any type annotations
        argument_names = [arg.strip().split(':')[0] for arg in arguments]

        function_info = {
            'name': function_name,
            'args': argument_names
        }
        functions.append(function_info)

    return functions

class FunctionArg:
    def __init__(self, name, type = None) -> None:
        self.name = name
        self.type = type

class FunctionSandbox:
    def __init__(self, code) -> None:
        self.code = code
        self.functions = extract_function_info(self.code)[0]
        self.name = self.functions['name']
        self.args = [FunctionArg(arg) for arg in self.functions['args']]

    def call(self, *args, **kwargs):
        return None