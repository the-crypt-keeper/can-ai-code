import time
import json
from pathlib import Path
from modal import Image, Stub, method, gpu
from huggingface_hub import snapshot_download

def save_meta(name, base, safetensors = True, bits = 4, group = 128, actorder = True, eos = ['<s>', '</s>']):
    with open("/model/_info.json",'w') as f:
        json.dump({
            "model_name": name,
            "model_base": base,
            "model_safetensors": safetensors,
            "model_bits": bits,
            "model_group": group,
            "model_actorder": actorder,
            "model_eos": eos,
        }, f)

def download_llama_30b_128g_v2():   
    MODEL_NAME = "Neko-Institute-of-Science/LLaMA-30B-4bit-128g"
    MODEL_BASE = "llama-30b-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, bits=4, group=128, actorder=True)

def download_koala_13b_v2():   
    MODEL_NAME = "TheBloke/koala-13B-GPTQ-4bit-128g"
    MODEL_BASE = "koala-13B-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+".safetensors"])
    save_meta(MODEL_NAME, MODEL_BASE, bits=4, group=128, actorder=True)

stub = Stub(name='exllama-v2')
stub.gptq_image = (
    Image.from_dockerhub(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential",
        ],
    )
    .run_commands(
        "git clone https://github.com/turboderp/exllama /repositories/exllama",
        "cd /repositories/exllama && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "cd /repositories/exllama && pip install safetensors sentencepiece ninja huggingface_hub",
        gpu="any",
    )
    .run_function(download_koala_13b_v2)
)

# Entrypoint import trick for when inside the remote container
if stub.is_inside(stub.gptq_image):
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    import sys
    sys.path.insert(0, str(Path("/repositories/exllama")))
    print('Starting up...')
    import torch
    from model import ExLlama, ExLlamaCache, ExLlamaConfig
    from tokenizer import ExLlamaTokenizer
    from generator import ExLlamaGenerator

#### NOTE: SET GPU TYPE HERE ####
@stub.cls(image=stub.gptq_image, gpu=gpu.A10G(count=1), concurrency_limit=1, container_idle_timeout=300)
class ModalExLlama:
    def __enter__(self):

        self.info = json.load(open('/model/_info.json'))
        print('Remote model info:', self.info)

        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()

        tokenizer_model_path = "/model/tokenizer.model"
        model_config_path = "/model/config.json"
        model_path = "/model/"+self.info['model_base']+(".safetensors" if self.info['model_safetensors'] else ".pt")

        self.config = ExLlamaConfig(model_config_path)
        self.config.model_path = model_path
        # config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
        # config.matmul_method = ExLlamaConfig.MatmulMethod.QUANT_ONLY
        self.config.max_seq_len = 2048

        print('Loading model...')
        self.model = ExLlama(self.config)
        self.cache = ExLlamaCache(self.model)
        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.tokenizer = ExLlamaTokenizer(tokenizer_model_path)

    def params(self, temperature=0.7, repetition_penalty=1.0, top_k=-1, top_p=1.0, max_new_tokens=512, beams=1, beam_length=1, **kwargs):
        return {
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "beams": beams,
            "beam_length": beam_length
        }

    @method()
    def generate(self, prompt, params):
        generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        generator.settings = ExLlamaGenerator.Settings()
        generator.settings.temperature = params['temperature']
        generator.settings.top_k = params['top_k']
        generator.settings.top_p = params['top_p']
        generator.settings.min_p = 0
        generator.settings.token_repetition_penalty_max = params['repetition_penalty']
        generator.settings.token_repetition_penalty_sustain = 256
        generator.settings.token_repetition_penalty_decay = 128
        generator.settings.beams = params['beams']
        generator.settings.beam_length = params['beam_length']

        answer = generator.generate_simple(prompt, max_new_tokens = params['max_new_tokens'])

        answer = answer.replace(prompt, '')

        return answer, self.info

# For local testing, run `modal run -q interview-exllama-modal.py --input prepare_junior-dev_python.ndjson --params params/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str):
    from prepare import save_interview

    model = ModalExLlama()

    interview = [json.loads(line) for line in open(input)]
    params_json = json.load(open(params,'r'))
    params_model = model.params(**params_json)
    model_info = None

    results = []
    for question in interview:
        print(question['name'], question['language'])

        # generate the answer
        answer, info = model.generate.call(question['prompt'], params=params_model)

        # save for later
        if model_info is None:
            model_info = info
            print('Local model info:', model_info)
        
        print()
        print(answer)
        print()

        result = question.copy()
        result['answer'] = answer
        result['params'] = params_model
        result['model'] = info['model_name']
        results.append(result)

    save_interview(input, 'none', params, model_info['model_name'], results)