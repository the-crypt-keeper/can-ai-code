import base64
import re
import tempfile
import subprocess
import json
import os
import logging
from jinja2 import Template

module_dir = os.path.dirname(os.path.abspath(__file__))

def extract_function_info(language, input_string):
    if language == 'python':
        function_regex = r"def\s+(.*)\s*\((.*)\)(.*):"
    elif language == 'javascript':
        function_regex = r"function\**\s+(\S*)\s*\((.*)\)(.*){"
    elif language == 'javascript-arrow':
        function_regex = r"\s+(\S*)\s*=\s*\((.*)\)(.*)\s*=>\s*{"
    else:
        raise Exception("extract_function_info: Unsupported language")
    
    matches = re.findall(function_regex, input_string, re.MULTILINE)
    # Javascript has a second style, try that if the normal one doesnt work.
    if len(matches) == 0 and language == 'javascript':
        return extract_function_info('javascript-arrow', input_string)

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

def run_shell_command(command, stdout_only = False):
    try:
        # Run the shell command and capture its output
        result = subprocess.run(command, shell=True, capture_output=True)     
        stdout_utf8 = result.stdout.decode('utf-8', 'ignore')
        stderr_utf8 = result.stderr.decode('utf-8', 'ignore')

        # Get the captured output
        output = stdout_utf8.strip()

        if not stdout_only or (result != 0 and output == ''):
            output += stderr_utf8.strip()

        # Get the return value
        return_value = result.returncode

        # Return the output and return value
        return output, return_value

    except subprocess.CalledProcessError as e:
        # Handle any errors that occurred during command execution
        print("Error:", e)
        return None, e.returncode

def build_sandbox(language, logger=None):
    """Build a sandbox Docker image for the specified language"""
    logger = logger or logging.getLogger(f"sandbox-{language}")
    logger.info(f"Building {language} sandbox")
    build_out, build_code = run_shell_command(f"cd {module_dir} && docker build . -f Dockerfile.{language} -t sandbox-{language} -q")
    if build_code != 0:
        raise Exception("Error "+str(build_code)+" building sandbox docker image:" + build_out)
    return True

def start_sandbox(language, instance_id=0, logger=None):
    """Start a sandbox Docker container for the specified language and instance"""
    logger = logger or logging.getLogger(f"sandbox-{language}-{instance_id}")
    sandbox_name = f"sandbox-{language}-{instance_id}"
    
    logger.info(f"Launching {language} sandbox (instance {instance_id})") 
    # Use different port mappings for different instances if needed
    start_out, start_code = run_shell_command(f"docker run -d --name {sandbox_name} sandbox-{language}")
    if start_code != 0:
        raise Exception("Error "+str(start_code)+" launching sandbox docker image:" + start_out)
    return sandbox_name

def stop_sandbox(sandbox_name, logger=None):
    """Stop and remove a sandbox Docker container"""
    logger = logger or logging.getLogger(f"sandbox-{sandbox_name}")
    logger.info(f"Stopping sandbox {sandbox_name}")
    run_shell_command(f"docker rm -f {sandbox_name}")
    
class FunctionArg:
    def __init__(self, name, type = None) -> None:
        self.name = name
        self.type = type

class FunctionSandbox:
    def __init__(self, code, language, instance_id=0, logger=None) -> None:
        self.code = code
        self.language = language
        self.instance_id = instance_id
        self.sandbox_name = f"sandbox-{language}-{instance_id}"
        self.logger = logger or logging.getLogger(f"sandbox-{language}-{instance_id}")

        try:
           self.functions = extract_function_info(self.language, self.code)[0]
        except:
           self.functions = { 'name': '', 'args': [] }
        self.name = self.functions['name']
        self.args = [FunctionArg(arg) for arg in self.functions['args']]

    def build_args(self, args):
        return_args = ''
        for i, arg in enumerate(args):
            if i != 0:
                return_args += ','
            return_args += json.dumps(arg)
            
        return return_args

    def call(self, *args, **kwargs):
        with open(module_dir+'/eval.'+self.language+'.tpl') as f:
            template = Template(f.read())

        script = template.render(call=self.name+'('+self.build_args(args)+')')
        
        script_b64 = base64.b64encode(script.encode('utf-8')).decode('utf-8')
        answer_b64 = base64.b64encode(self.code.encode('utf-8')).decode('utf-8')

        if self.language == "python":
            output, value = run_shell_command(f"docker exec -i -e WRAPPER_SOURCE={script_b64} -e ANSWER_SOURCE={answer_b64} {self.sandbox_name} /timeout.sh python /wrapper", stdout_only=True)
        elif self.language == "javascript":
            output, value = run_shell_command(f"docker exec -i -e WRAPPER_SOURCE={script_b64} -e ANSWER_SOURCE={answer_b64} {self.sandbox_name} /timeout.sh node /wrapper", stdout_only=True)
       
        start_index = output.find("###")
        if start_index == -1:
            if value != 0:
                return { "error": "non-zero result code "+str(value), "output": output }
            else:
                return output
                
        rv_text = output[start_index + 3:].strip()
        #print(rv_text)
        return json.loads(rv_text)
