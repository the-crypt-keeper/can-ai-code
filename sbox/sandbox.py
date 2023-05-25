import re
import tempfile
import subprocess
import json
import os
from jinja2 import Template

module_dir = os.path.dirname(os.path.abspath(__file__))

def extract_function_info(input_string):
    function_regex = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\):"
    matches = re.findall(function_regex, input_string, re.MULTILINE)

    functions = []
    for match in matches:
        function_name = match[0]
        arguments = match[1].split(',')

        # Extract argument names by removing any type annotations
        argument_names = [arg.strip().split(':')[0] for arg in arguments if arg]

        function_info = {
            'name': function_name,
            'args': argument_names
        }
        functions.append(function_info)

    return functions

def run_shell_command(command):
    try:
        # Run the shell command and capture its output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Get the captured output
        output = result.stdout.strip()

        if output == '':
            output = result.stderr.strip()

        # Get the return value
        return_value = result.returncode

        # Return the output and return value
        return output, return_value

    except subprocess.CalledProcessError as e:
        # Handle any errors that occurred during command execution
        print("Error:", e)
        return None, e.returncode
    
class FunctionArg:
    def __init__(self, name, type = None) -> None:
        self.name = name
        self.type = type

class FunctionSandbox:
    def __init__(self, code) -> None:
        self.code = code
        try:
           self.functions = extract_function_info(self.code)[0]
        except:
           self.functions = { 'name': '', 'args': [] }
        self.name = self.functions['name']
        self.args = [FunctionArg(arg) for arg in self.functions['args']]

        build_out, build_code = run_shell_command('cd '+module_dir+' && docker build . -f Dockerfile.python -t sandbox-py -q')
        if build_code != 0:
            raise Exception("Error building docker image:" + build_out)

    def call(self, *args, **kwargs):
        output = None
        with open(module_dir+'/eval.py.tpl') as f:
            template = Template(f.read())
            output = template.render(name=self.name, args=self.args, kwargs=kwargs)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            script_file = temp_file.name
            temp_file.write(template.render(call=self.name+'('+','.join([str(x) for x in args])+')'))

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            answer_file = temp_file.name
            temp_file.write(self.code)

        output, value = run_shell_command('docker run -it -v '+script_file+':/wrapper.py -v '+answer_file+':/answer.py sandbox-py python /wrapper.py')
        
        start_index = output.find("###")
        if start_index == -1:
            return output
        
        rv_text = output[start_index + 3:].strip()
        return json.loads(rv_text)
