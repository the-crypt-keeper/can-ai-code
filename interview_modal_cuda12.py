from modal import App, Image, method, enter, gpu, Secret, Mount
from huggingface_hub import snapshot_download
import json
from jinja2 import Template
from interview_cuda import *

def download_model(name, info = {}, **kwargs):
    for k,v in kwargs.items(): info[k] = v
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name, **info}, f)
    snapshot_download(name, **kwargs)

# LLAMA2 7B
def model_llama_chat_7b_e8p(): download_model('relaxml/Llama-2-7b-chat-E8P-2Bit')
# Mistral 7B
def model_hermes2_pro_mistral_7b(): download_model('NousResearch/Hermes-2-Pro-Mistral-7B')
def model_ajibawa2023_code_mistral_7b(): download_model('ajibawa-2023/Code-Mistral-7B')
# Starcoder2
def model_dolphincoder_starcoder2_7b(): download_model('cognitivecomputations/dolphincoder-starcoder2-7b')
def model_dolphincoder_starcoder2_15b(): download_model('cognitivecomputations/dolphincoder-starcoder2-15b')
# LLama3 8B
def model_llama3_instruct_8b(): download_model('meta-llama/Meta-Llama-3-8B-Instruct')
def model_llama3_instruct_8b_awq(): download_model('casperhansen/llama-3-8b-instruct-awq', info={'eos_token_id': 128009})
def model_llama3_instruct_8b_gptq_8bpw(): download_model('astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit', info={'eos_token_id': 128009})
def model_llama3_instruct_8b_gptq_4bpw(): download_model('MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ', info={'eos_token_id': 128009})
def model_llama3_instruct_8b_exl2_6bpw(): download_model('turboderp/Llama-3-8B-Instruct-exl2', revision='6.0bpw', info={'eos_token_id': 128009})
def model_ajibawa_code_llama3(): download_model('ajibawa-2023/Code-Llama-3-8B')
def model_rombodawg_llama3_8b_instruct_coder(): download_model('rombodawg/Llama-3-8B-Instruct-Coder')

# LLama3 70B
def model_llama3_instruct_70b_exl2_4bpw(): download_model('turboderp/Llama-3-70B-Instruct-exl2', revision='4.0bpw', info={'eos_token_id': 128009})
def model_llama3_instruct_70b_gptq(): download_model('MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ', info={'eos_token_id': 128009})
# CodeQwen
def model_codeqwen_7b_awq(): download_model("Qwen/CodeQwen1.5-7B-Chat-AWQ")
def model_codeqwen_7b_fp16(): download_model("Qwen/CodeQwen1.5-7B-Chat")
# ibm-granite
def model_granite_20b(): download_model("ibm-granite/granite-20b-code-instruct")
def model_granite_34b(): download_model("ibm-granite/granite-34b-code-instruct")
# starcoder2
def model_starcoder2_instruct_0p1(): download_model('bigcode/starcoder2-15b-instruct-v0.1', info={'eos_token_id': 0})
def model_starchat2_0p1(): download_model('HuggingFaceH4/starchat2-15b-v0.1', info={'eos_token_id': 49153})
def modal_starchat2_0p1_awq(): download_model('stelterlab/starchat2-15b-v0.1-AWQ', info={'eos_token_id': 49153})
def model_starchat2_sft_0p1(): download_model('HuggingFaceH4/starchat2-15b-sft-v0.1', info={'eos_token_id': 49153})
# Mixtral
def model_mixtral_8x22b_instruct_awq(): download_model('MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ')
def model_mixtral_8x22b_instruct_gptq(): download_model('jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit')
def model_mixtral_8x22b_instruct_exl2(): download_model('turboderp/Mixtral-8x22B-Instruct-v0.1-exl2', revision='4.0bpw')
def model_wizardlm2_8x22b_awq(): download_model('MaziyarPanahi/WizardLM-2-8x22B-AWQ')
def model_wizardlm2_8x22b_exl2(): download_model('Dracones/WizardLM-2-8x22B_exl2_4.0bpw')

##### SELECT RUNTIME HERE #############
#RUNTIME = "transformers"
#QUANT = QUANT_FP16
#RUNTIME = "ctranslate2"
#RUNTIME = "vllm"
#RUNTIME = "autogptq"
#RUNTIME = "exllama"
RUNTIME = "exllama2-th"
#RUNTIME = "awq"
#RUNTIME = "quipsharp"
#######################################

##### SELECT GPU HERE #################
#gpu_request = gpu.T4(count=1)
#gpu_request = gpu.A10G(count=1)
gpu_request = gpu.A100(count=1, memory=80)
#######################################

vllm_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04",
                        setup_dockerfile_commands=["RUN apt-get update", "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential"])
    .pip_install(
        "transformers==4.40.0",
        "optimum==1.19.1",
        "tiktoken==0.6.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.29.2",
        "einops==0.6.1",
        "sentencepiece==0.1.99",
        "hf-transfer~=0.1",
        "scipy==1.10.1",
        "pyarrow==11.0.0",
        "protobuf==3.20.3",
        "vllm==0.4.1",
        "auto-gptq==0.7.1",
        "https://github.com/turboderp/exllamav2/releases/download/v0.0.19/exllamav2-0.0.19+cu121-cp310-cp310-linux_x86_64.whl"
    )
    .pip_install("flash-attn==2.5.7") # this errors out unless torch is already installed
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    ##### SELECT MODEL HERE ##############    
    .run_function(model_mixtral_8x22b_instruct_exl2, secrets=[Secret.from_name("my-huggingface-secret")])
    ######################################
)
app = App(image=vllm_image)

@app.cls(gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300, secrets=[Secret.from_name("my-huggingface-secret")], mounts=[Mount.from_local_python_packages("interview_cuda")])
class ModalWrapper:
    @enter()
    def startup(self):
        self.info = json.load(open('./_info.json'))

        if RUNTIME == "transformers":
            self.wrapper = InterviewTransformers(self.info['model_name'], self.info, quant=QUANT)
        elif RUNTIME == "vllm":
            gpu_split = 2 if gpu_request.count > 1 else None
            print(gpu_split)
            self.wrapper = InterviewVLLM(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "autogptq":
            self.wrapper = InterviewAutoGPTQ(self.info['model_name'], self.info)
        elif RUNTIME == "exllama":
            gpu_split = '17,24' if gpu_request.count > 1 else None
            self.wrapper = InterviewExllama(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME[0:8] == "exllama2":
            token_healing = '-th' in RUNTIME
            cache_8bit = '8b' in RUNTIME
            self.wrapper = InterviewExllama2(self.info['model_name'], self.info, token_healing=token_healing, cache_8bit=cache_8bit)
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
            raise Exception("Unknown RUNTIME")

        self.wrapper.load()

    @method()
    def generate(self, prompt, params):
        return self.wrapper.generate(prompt, params)

# For local testing, run `modal run -q interview_modal.py --input results/prepare.ndjson --params params/precise.json`
@app.local_entrypoint()
def main(input: str, params: str, iterations: int = 1, templateout: str = "", batch: bool = False):
    from prepare import save_interview
    from interview_cuda import interview_run

    output_template = Template(open(templateout).read()) if templateout else None

    tasks = []
    for param_file in params.split(','):
        for input_file in input.split(','):
            if param_file != '' and input_file != '':
                tasks.append((param_file, input_file))

    model = ModalWrapper()

    for param_file, input_file in tasks:
      interview = [json.loads(line) for line in open(input_file)]
      params_json = json.load(open(param_file,'r'))

      for iter in range(iterations):
        print(f"Starting iteration {iter} of {param_file} {input_file}")
        results, remote_info = interview_run(RUNTIME, model.generate.remote, interview, params_json, output_template, batch=batch )
        save_interview(input_file, templateout if templateout else 'none', param_file, remote_info['model_name'], results)
