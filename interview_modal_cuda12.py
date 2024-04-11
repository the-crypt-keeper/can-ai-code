from modal import Stub, Image, method, enter, gpu, Secret, Mount
from huggingface_hub import snapshot_download
import time
import json
from jinja2 import Template
from interview_cuda import *

def download_model(name, info = {}, **kwargs):
    for k,v in kwargs.items(): info[k] = v
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name, **info}, f)
    snapshot_download(name, **kwargs)

def model_llama_chat_7b_e8p(): download_model('relaxml/Llama-2-7b-chat-E8P-2Bit')
def model_hermes2_pro_mistral_7b(): download_model('NousResearch/Hermes-2-Pro-Mistral-7B')

image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04",
                        setup_dockerfile_commands=["RUN apt-get update", "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential"])
    .pip_install(
        "transformers==4.39.3",
        "optimum==1.18.1",
        "tiktoken==0.6.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.29.2",
        "einops==0.6.1",
        "sentencepiece==0.1.99",
        "hf-transfer~=0.1",
        "scipy==1.10.1",
        "pyarrow==11.0.0",
        "protobuf==3.20.3",
        "vllm==0.4.0.post1",
        "auto-gptq==0.7.1"        
    )  
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) #, "GITHUB_ACTIONS": "true", "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.9 9.0"})
    # .pip_install(
    #     "vllm==0.4.0.post1",
    #     "auto-gptq==0.7.1",
    #     # "exllamav2==0.0.18"
    # )
    # .run_commands(
    #     "git clone https://github.com/Cornell-RelaxML/quip-sharp.git /repositories/quip-sharp && cd /repositories/quip-sharp && git checkout 1d6e3c2d4c144eba80b945cca5429ce8d79d2cec && pip install -r requirements.txt && cd quiptools && python setup.py install"
    # )
    # .pip_install("git+https://github.com/NVIDIA/TransformerEngine.git@main")
    
    ##### SELECT MODEL HERE ##############    
    .run_function(model_hermes2_pro_mistral_7b, secrets=[Secret.from_name("my-huggingface-secret")])
    ######################################
)
stub = Stub(image=image)

##### SELECT RUNTIME HERE #############
#RUNTIME = "transformers"
#QUANT = QUANT_FP16
#RUNTIME = "ctranslate2"
RUNTIME = "vllm"
#RUNTIME = "autogptq"
#RUNTIME = "exllama"
#RUNTIME = "exllama2"
#RUNTIME = "awq"
#RUNTIME = "quipsharp"
#######################################

##### SELECT GPU HERE #################
#gpu_request = gpu.T4(count=1)
gpu_request = gpu.A10G(count=1)
#gpu_request = gpu.A100(count=1)
#######################################

@stub.cls(gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300, secrets=[Secret.from_name("my-huggingface-secret")], mounts=[Mount.from_local_python_packages("interview_cuda")])
class ModalWrapper:
    @enter()
    def startup(self):
        self.info = json.load(open('./_info.json'))

        if RUNTIME == "transformers":
            self.wrapper = InterviewTransformers(self.info['model_name'], self.info, quant=QUANT)
        elif RUNTIME == "vllm":
            gpu_split = 2 if gpu_request.count == 2 else None
            print(gpu_split)
            self.wrapper = InterviewVLLM(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "autogptq":
            self.wrapper = InterviewAutoGPTQ(self.info['model_name'], self.info)
        elif RUNTIME == "exllama":
            gpu_split = '17,24' if gpu_request.count == 2 else None
            self.wrapper = InterviewExllama(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "exllama2":
            #
            self.wrapper = InterviewExllama2(self.info['model_name'], self.info)
        elif RUNTIME == "awq":
            if self.info.get('big_model'):
                gpu_split = '0,1' if gpu_request.count == 2 else '0,cpu'
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
@stub.local_entrypoint()
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
