#!/usr/bin/env python3
import fire
import json
import sys
from jinja2 import Template
import subprocess

def main(model: str, runtime: str, gpu: str = "A10G", input: str = "", interview: str = "senior", prompt:str="", params: str = "", templateout: str = "", revision: str = "", info: str = "{}", quant: str = "fp16", context : int = 2048):
    model_info = json.loads(info) if isinstance(info, str) else info
    
    model_args = { 'info': model_info }
    model_args['info']['quant'] = quant
    model_args['info']['context_size'] = context
    
    if revision: model_args['revision'] = revision
    if isinstance(revision, int): raise Exception("Please escape --revision with \\' to avoid Fire parsing issues.")
    model_clean = model.replace('/','-').replace('_','-').replace('.','-')
    model_clean_py = model_clean.replace('-','_')
    
    if input == "" and interview == "": raise Exception("Please provide either --input or --interview")
   
    input_template = "interview_modal.tpl.py"
    tpl = Template(open(input_template).read())
    
    modal_params = {
        'MODELSLUG': model_clean_py, 
        'MODELARGS': str(model_args),
        'MODELNAME': model,
        'RUNTIME': runtime,
        'GPUREQUEST': f'''"{gpu}"'''
    }
    
    output = tpl.render(modal_params)
    output_script = f"modal_run_{model_clean_py}_{runtime}_{gpu}.py"
    with open(output_script,'w') as f:
        f.write(output)
    
    args = []
    if input: args += ["--input",input]
    if interview: args += ["--interview",interview]
    if params: args += ["--params", params]
    if templateout: args += ["--templateout",templateout]
    if prompt: args += ["--prompt",prompt]
    if runtime == "vllm": args += ["--batch"]

    print(f"Rendered {output_script} with {modal_params}, executing via modal run with {args}")
    subprocess.run(["modal", "run", output_script]+args)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
