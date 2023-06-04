import time
import json
from pathlib import Path
from modal import Image, Stub, method, create_package_mounts, gpu

#### SEE NOTE BELOW! ####
MODEL_NAME = "tsumeone/llama-30b-supercot-4bit-cuda"
MODEL_BASE = "4bit"
MODEL_FILES = ["*.json","*.model",MODEL_BASE+"*"]
MODEL_SAFETENSORS = True
MODEL_BITS = 4
MODEL_GROUP = -1
MODEL_ACTORDER = True

stub = Stub(name=MODEL_NAME.replace('/', '-'))

#### NOTE: Modal will not rebuild the container unless this function name or it's code contents change.
####       It is NOT sufficient to change any of the constants above.
####       Suggestion is to rename this function after the model.
def download_llama30b_nogroup_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        local_dir=Path("/model"),
        repo_id=MODEL_NAME,
        allow_patterns=MODEL_FILES
    )

stub.gptq_image = (
    Image.from_dockerhub(
        "nvidia/cuda:11.7.1-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential",
        ],
    )
    .run_commands(
        "git clone https://github.com/PanQiWei/AutoGPTQ /repositories/AutoGPTQ",
        "cd /repositories/AutoGPTQ && pip install . && pip install einops sentencepiece && python setup.py install",
        gpu="any",
    )
    .run_function(download_llama30b_nogroup_model)
)

if stub.is_inside(stub.gptq_image):
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    import sys
    sys.path.insert(0, str(Path("/repositories/AutoGPTQ")))
    import torch
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM
    from auto_gptq.modeling import BaseQuantizeConfig

#### NOTE: SET GPU TYPE HERE ####
@stub.cls(image=stub.gptq_image, gpu=gpu.A10G(count=1), concurrency_limit=1, container_idle_timeout=300)
class ModalGPTQ:
    def __enter__(self):
        quantized_model_dir = "/model"
        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False)

        quantize_config = BaseQuantizeConfig()
        quantize_config.desc_act = MODEL_ACTORDER
        quantize_config.bits = MODEL_BITS
        quantize_config.group = MODEL_GROUP

        print('Loading model...')
        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, model_basename=MODEL_BASE, device_map="auto", load_in_8bit=True, use_triton=False, use_safetensors=MODEL_SAFETENSORS, torch_dtype=torch.float32, trust_remote_code=True, quantize_config=quantize_config)
        
        self.model = model
        self.tokenizer = tokenizer
        print(f"Model loaded in {time.time() - t0:.2f}s")

    def params(self, temperature=0.7, repetition_penalty=1.0, top_k=-1, top_p=1.0, max_new_tokens=512, **kwargs):
        return {
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens
        }

    @method()
    def generate(self, prompt, params):
        tokens = self.tokenizer(prompt, return_tensors="pt").to("cuda:0").input_ids
        output = self.model.generate(input_ids=tokens, do_sample=True, **params)
        return self.tokenizer.decode(output[0])

# For local testing, run `modal run -q interview-gptq-modal.py --input questions.csv --params model_parameters/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str):
    from prepare import save_interview

    model = ModalGPTQ()

    interview = [json.loads(line) for line in open(input)]
    params_json = json.load(open(params,'r'))
    params_model = model.params(**params_json)

    results = []
    for question in interview:
        print(question['name'], question['language'])

        answer = model.generate.call(question['prompt'], params=params_model)

        print()
        print(answer)
        print()

        result = question.copy()
        result['answer'] = answer
        result['params'] = params_model
        result['model'] = MODEL_BASE
        results.append(result)

    save_interview(input, 'none', params, MODEL_NAME, results)