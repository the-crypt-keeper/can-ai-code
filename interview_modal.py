from modal import Stub, Image, method, gpu, Secret, create_package_mounts
from huggingface_hub import snapshot_download
import time
import json
from jinja2 import Template

def download_model(name, **kwargs):
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name}, f)
    snapshot_download(name, **kwargs)

def download_codegen_2p5_7b_mono_model():
    download_model("Salesforce/codegen25-7b-mono")

def download_codegen_2p5_7b_multi_model():
    download_model("Salesforce/codegen25-7b-multi")

def download_codegen_2_1b_multi_model():
    download_model("Salesforce/codegen2-1B")

def download_codegen_2_3p7b_multi_model():
    download_model("Salesforce/codegen2-3_7B")

def download_codegen_2_7b_multi_model():
    download_model("Salesforce/codegen2-7B")

def download_falcon_instruct_7b_model():
    download_model("tiiuae/falcon-7b-instruct", allow_patterns=["*.json","*.model","pytorch*.bin"])

def download_replit_code_instruct_3b_model():
    download_model("sahil2801/replit-code-instruct-glaive")

def download_replit_code_v1_3b_model():
    download_model("replit/replit-code-v1-3b")

def download_vicuna_1p3_7b_model():
    download_model("lmsys/vicuna-7b-v1.3")

def download_llama2_7b_model():
    download_model("meta-llama/Llama-2-7b-hf", ignore_patterns=["*.bin"])

def download_llama2_13b_model():
    download_model("meta-llama/Llama-2-13b-hf", ignore_patterns=["*.bin"])

def download_redmond_puffin_preview_13b_model():
    download_model("NousResearch/Redmond-Puffin-13B")

def download_codeCherryPop_7b_model():
    download_model("TokenBender/llama2-7b-chat-hf-codeCherryPop-qLoRA-merged")

def download_tinycoderpy_model():
    download_model("bigcode/tiny_starcoder_py")

image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:23.06-py3")
    .pip_install(
        "transformers==4.30.2",
        "tiktoken==0.4.0",
        "bitsandbytes==0.40.1.post1",
        "accelerate==0.21.0"
    )
    .pip_install("einops==0.6.1", "sentencepiece==0.1.99")
    .run_function(download_tinycoderpy_model, secret=Secret.from_name("my-huggingface-secret"))
)
stub = Stub(image=image)

gpu_request = gpu.A10G(count=1)
@stub.cls(gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300, secret=Secret.from_name("my-huggingface-secret"), mounts=create_package_mounts(["interview_cuda"]))
class ModalWrapper:
    def __enter__(self):
        from interview_cuda import InterviewTransformers
        self.info = json.load(open('./_info.json'))
        self.wrapper = InterviewTransformers(self.info['model_name'], self.info)
        self.wrapper.load()

    @method()
    def generate(self, prompt, params):
        return self.wrapper.generate(prompt, params)
    
# For local testing, run `modal run -q interview-transformers-modal.py --input questions.csv --params model_parameters/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str, iterations: int = 1, templateout: str = ""):
    from prepare import save_interview
    from interview_cuda import interview_run

    model = ModalWrapper()

    interview = [json.loads(line) for line in open(input)]
    params_json = json.load(open(params,'r'))
    output_template = Template(open(templateout).read()) if templateout else None

    for iter in range(iterations):
        results, remote_info = interview_run(model.generate.call, interview, params_json, output_template)
        save_interview(input, templateout if templateout else 'none', params, remote_info['model_name'], results)
