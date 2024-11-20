import fire
import json
import sys
from jinja2 import Template
import subprocess

GPU_STRINGS = {
    "T4": "modal.gpu.T4(count=1)",
    "A10": "modal.gpu.A10G(count=1)",
    "A10x2": "modal.gpu.A10G(count=2)",
    "A100-80": "modal.gpu.A100(count=1, memory=80)",
    "A100-40": "modal.gpu.A100(count=1, memory=40)",
}

def main(model: str, runtime: str, gpu: str = "A10", input: str = "", interview: str = "senior", params: str = "", templateout: str = "", revision: str = "", info: str = "{}"):
    model_args = { 'info': json.loads(info) }
    if revision: model_args['revision'] = revision
    if isinstance(revision, int): raise Exception("Please escape --revision with \\' to avoid Fire parsing issues.")
    model_clean = model.replace('/','-').replace('_','-')
    model_clean_py = model_clean.replace('-','_')
    
    if gpu not in GPU_STRINGS.keys():
        raise Exception("Please provide valid --gpu: "+",".join(GPU_STRINGS.keys()))
    
    input_template = "interview_modal.tpl.py"
    tpl = Template(open(input_template).read())
    
    modal_params = {
        'MODELSLUG': model_clean_py, 
        'MODELARGS': str(model_args),
        'MODELNAME': model,
        'RUNTIME': runtime,
        'GPUREQUEST': GPU_STRINGS[gpu]
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
    if runtime == "vllm": args += ["--batch"]

    print(f"Rendered {output_script} with {modal_params}, executing via modal run with {args}")
    subprocess.run(["modal", "run", "-q", output_script]+args)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
