import fire
import json
import sys
from jinja2 import Template
import subprocess

# Format: <gpu>-<memory>x<count>
# Examples: T4, A10Gx2, A100-40x4
def parse_gpu_string(gstr):
    count = 1
    memory = None
    
    size_split = gstr.split('x')
    if len(size_split) > 1: count = size_split[1]
    mem_split = size_split[0].split('-')
    if len(mem_split) > 1: memory = mem_split[1]
    
    return f"modal.gpu.{mem_split[0]}(count={count}" + (f", size='{memory}')" if memory else ")")

def main(model: str, runtime: str, gpu: str = "A10G", input: str = "", interview: str = "senior", prompt:str="", params: str = "", templateout: str = "", revision: str = "", info: str = "{}"):
    model_args = { 'info': json.loads(info) }
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
        'GPUREQUEST': parse_gpu_string(gpu)
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
