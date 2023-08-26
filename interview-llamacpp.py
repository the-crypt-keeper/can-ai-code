#!/usr/bin/env python3
import json
import tempfile
from sys import exit
from pathlib import Path
from sbox.sandbox import run_shell_command
from prepare import save_interview
from jinja2 import Template
from copy import copy
from interview_cuda import interview_run

class InterviewLlamaCpp:
    def __init__(self, model_name, model_info = {}, quant = None, gpu_split = None):
        self.model_name = Path(model_name).stem
        self.info = model_info
        self.info['model_name'] = self.model_name

        self.batch = False

        self.model = model_name
        self.threads = self.info.get('threads', 16)
        self.main = self.info.get('main', '/llama/main')
        self.args = self.info.get('args', '')
        self.ssh = self.info.get('ssh', '')

        self.stop_seq = self.info.get('generate_args', {}).get('stop_seq', [])
  
    def load(self):
        pass

    def build_llama_command(self, params):
        param_map = {
            'n_predict': 'max_new_tokens',
            'temp': 'temperature',
            'top_k': 'top_k',
            'top_p': 'top_p',
            'repeat_last_n': 'repeat_last_n',
            'repeat_penalty': 'repetition_penalty',
            'mirostat': 'mirostat',
            'mirostat-lr': 'mirostat-lr',
            'mirostat-ent': 'mirostat-ent'
        }

        if self.main.find('starcoder') > -1:
            param_map['repeat-penalty'] = 'repetition_penalty'
            param_map['repeat-last-n'] = 'repeat_last_n'
            del param_map['repeat_last_n']
            del param_map['repeat_penalty']

        llama_command = f"{self.main} {self.args} --threads {self.threads} --model {self.model}"

        for seq in self.stop_seq:
            eseq = seq.replace('\n','\\n')
            llama_command += f" -r $'{eseq}'"

        for k,v in param_map.items():
            if v in params:
                llama_command += f' --{k} {params[v]}'

        return llama_command

    def generate(self, prompt, params):
        llama_command = self.build_llama_command(params)

        prompt_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        prompt_file.write(prompt)
        prompt_file.close()

        cmdline = llama_command + f' --file {prompt_file.name}'

        if self.ssh:
            scp_command = f"scp {prompt_file.name} {self.ssh}:{prompt_file.name}"
            print('Copying to remote machine:', scp_command)

            output, rv = run_shell_command(scp_command)
            if rv != 0:
                print('Failed to copy to remote machine:', output)
                exit(1)

            cmdline = f"ssh {self.ssh} '{cmdline}'"

        print('Executing llama.cpp: '+cmdline)

        answer, rv = run_shell_command(cmdline, stdout_only=True)
        if rv != 0:
            print('Failed to execute:', output)
            exit(1)

        # remove prompt from answer
        start_offset = max(answer.rfind(prompt), 0)
        start_offset += len(prompt)
        answer = answer[start_offset:]

        # for starcoder remove the trailer
        end_offset = answer.find('\nmain: mem per token =')
        if end_offset > -1:
            answer = answer[:end_offset]

        # remove any eos/eot tokens
        for eos in ['<|endoftext|>', '<|end|>', '<|end_of_turn|>', *self.stop_seq]:
            answer = answer.replace(eos, '')

        # return result
        info_copy = copy(self.info)
        info_copy['sampling_params'] = cmdline
        return answer, info_copy

def cli(input: str, model:str, params: str, templateout: str = "", iterations: int=1, info: str = "{}", main: str = "/llama/main", threads: int = 16, ssh: str = ""):

    info_cmdline = json.loads(info) if isinstance(info, str) else info
    info_cmdline['main'] = main
    info_cmdline['threads'] = threads
    info_cmdline['ssh'] = ssh

    llama = InterviewLlamaCpp(model, info_cmdline)
    llama.load()

    tasks = []
    for param_file in params.split(','):
        for input_file in input.split(','):
            tasks.append((param_file, input_file))

    for param_file, input_pairs in tasks:
      insplit = input_pairs.split(':')
      input_file = insplit[0]
      templateout_file = insplit[1] if len(insplit)>1 else templateout

      interview = [json.loads(line) for line in open(input_file)]
      output_template = Template(open(templateout_file).read()) if templateout_file else None
      params_json = json.load(open(param_file,'r'))

      for iter in range(iterations):
        print("Starting", llama.model_name, "iter=", iter, "param_file=", param_file, "input_file=", input_file, "templateout_file=", templateout_file)
        results, remote_info = interview_run("llamacpp", llama.generate, interview, params_json, output_template, batch=llama.batch)
        save_interview(input_file, templateout_file if templateout_file else 'none', param_file, remote_info['model_name'], results)

if __name__ == "__main__":
    import fire
    fire.Fire(cli)