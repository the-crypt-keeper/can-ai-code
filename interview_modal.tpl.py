from huggingface_hub import snapshot_download
import json
from jinja2 import Template
import modal
from interview_cuda import *

def download_model(name, info = {}, **kwargs):
    for k,v in kwargs.items(): info[k] = v
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name, **info}, f)
    snapshot_download(name, ignore_patterns=["*.pth"], **kwargs)

# Example: Gemma2
#def model_gemma2_9b_instruct(): download_model('google/gemma-2-9b-it', info={'generate_args': { 'stop_seq': ['**Explanation:**']}})
#def model_gemma2_27b_instruct(): download_model('google/gemma-2-27b-it', info={'generate_args': { 'stop_seq': ['**Explanation:**']}})
#def model_gemma2_27b_gptq(): download_model('ModelCloud/gemma-2-27b-it-gptq-4bit', info={'VLLM_ATTENTION_BACKEND': 'FLASHINFER', 'generate_args': { 'stop_seq': ['**Explanation:**']}})
#def model_gemma2_2b_instruct(): download_model('google/gemma-2-2b-it', info={'generate_args': { 'stop_seq': ['**Explanation:**']}})
# Example: llama3.1
#def model_llama31_8b_instruct(): download_model('meta-llama/Meta-Llama-3.1-8B-Instruct')
#def model_llama31_8b_instruct_hqq(): download_model('mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib')
#def model_llama31_8b_exl2_8bpw(): download_model('turboderp/Llama-3.1-8B-Instruct-exl2', revision='8.0bpw', info={'eos_token_id': 128009})

model_args = {{MODELARGS}}
def model_{{MODELSLUG}}():
    download_model("{{MODELNAME}}", **model_args)

##### SELECT RUNTIME HERE #############
QUANT = QUANT_FP16

#RUNTIME = "transformers"
#RUNTIME = "ctranslate2"
#RUNTIME = "vllm"
#RUNTIME = "autogptq"
#RUNTIME = "exllama2"
#RUNTIME = "awq"
#RUNTIME = "quipsharp"
#RUNTIME = "hqq"

RUNTIME = "{{RUNTIME}}"
#######################################

##### SELECT GPU HERE #################
#gpu_request = modal.gpu.T4(count=1)
#gpu_request = modal.gpu.A10G(count=1)
#gpu_request = modal.gpu.A100(count=1, memory=80)

gpu_request = {{GPUREQUEST}}
#######################################

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04",
                        setup_dockerfile_commands=["RUN apt-get update", "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential"])
    .pip_install(
        "transformers==4.46.0",
        "tiktoken==0.7.0",
        "bitsandbytes==0.44.1",
        "accelerate==1.0.1",
        "einops==0.6.1",
        "sentencepiece==0.2.0",
        "hf-transfer~=0.1",
        "scipy==1.10.1",
        "pyarrow==11.0.0",
        "protobuf==3.20.3",
        "vllm==0.6.3.post1",
        "auto-gptq==0.7.1",
        "https://github.com/turboderp/exllamav2/releases/download/v0.2.3/exllamav2-0.2.3+cu121.torch2.3.1-cp310-cp310-linux_x86_64.whl"
    )
    .pip_install("flash-attn==2.6.3") # this errors out unless torch is already installed
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    ##### SELECT MODEL HERE ##############    
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .run_function(model_{{MODELSLUG}},
                  secrets=[modal.Secret.from_name("my-huggingface-secret")])
    ######################################
)
app = modal.App(image=vllm_image)

@app.cls(gpu=gpu_request, concurrency_limit=1, timeout=600, secrets=[modal.Secret.from_name("my-huggingface-secret")])
class ModalWrapper:
    @modal.enter()
    def startup(self):
        self.info = json.load(open('./_info.json'))

        if RUNTIME == "transformers":
            self.wrapper = InterviewTransformers(self.info['model_name'], self.info, quant=QUANT)
        elif RUNTIME == "vllm":
            gpu_split = gpu_request.count if gpu_request.count > 1 else None
            self.wrapper = InterviewVLLM(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "autogptq":
            self.wrapper = InterviewAutoGPTQ(self.info['model_name'], self.info)
        elif RUNTIME == "hqq":
            self.wrapper = InterviewHQQ(self.info['model_name'], self.info)
        elif RUNTIME[0:8] == "exllama2":
            self.wrapper = InterviewExllama2(self.info['model_name'], self.info, token_healing=True, cache_4bit=False)
        elif RUNTIME == "awq":
            if self.info.get('big_model'):
                gpu_split = '0,1' if gpu_request.count > 1 else '0,cpu'
            else:
                gpu_split = None
            self.wrapper = InterviewAWQ(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "ctranslate2":
            self.wrapper = InterviewCtranslate2(self.info['model_name'], self.info)
        elif RUNTIME == "quipsharp":
            self.wrapper = InterviewQuipSharp(self.info['model_name'], self.info)
        else:
            print(f"Unknown RUNTIME")
            self.wrapper = None
        
        try:
            self.wrapper.load()
        except Exception as e:
            print(f'{RUNTIME} crashed during load:', e)
            self.wrapper = None
    
    @modal.method()
    def generate(self, prompt, params):
        return self.wrapper.generate(prompt, params)

@app.local_entrypoint()
def main(input: str = "", interview:str = "", params: str = "", templateout: str = "", batch: bool = False):
    from prepare import save_interview, prepare_interview
    from interview_cuda import interview_run
    from transformers import AutoTokenizer
    
    print("Launching modal enviroment for {{MODELNAME}} with {{RUNTIME}}...")

    output_template = Template(open(templateout).read()) if templateout else None
    if params == "": params = "params/greedy-hf.json" if RUNTIME == "transformers" else "params/greedy-openai.json"
    params_json = json.load(open(params,'r'))

    interviews = []       
    if input != "":
        for input_file in input.split(','):
            interview = [json.loads(line) for line in open(input_file)]
            interviews.append( (input_file, interview) )
            print(f"Loaded {len(interview)} questions from {input_file}.")
    elif interview != "":
        tokenizer = AutoTokenizer.from_pretrained("{{MODELNAME}}", trust_remote_code=True, revision=model_args.get('revision'))
        for interview_name in interview.split(','):
            language = "python,javascript"
            template_name = "chat-simple-"+"{{MODELSLUG}}".replace('_','-')
            message_template = [{'role': 'user', 'content': Template("Write a {language} function {Signature} {Input} that returns {Output}".replace('{','{'+'{').replace('}','}'+'}'))}]
            output_filename, interview = prepare_interview(interview_name, language, message_template, template_name, tokenizer)
            interviews.append( (output_filename, interview) )
            print(f"Expanded {len(interview)} questions from {interview_name}.")
    else:
        raise Exception("Please provide either --input or --interview")

    print("Init model...")
    model = ModalWrapper()

    for input_file, interview in interviews:
        print(f"Starting params={params} input={input_file}")
        results, remote_info = interview_run(RUNTIME, model.generate.remote, interview, params_json, output_template, batch=batch )
        save_interview(input_file, templateout if templateout else 'none', params, remote_info['model_name'], results)
