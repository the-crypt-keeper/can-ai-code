import time
import json
from pathlib import Path
from modal import Image, Stub, method, gpu
from huggingface_hub import snapshot_download

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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

def download_llama_65b_128g_v2():   
    MODEL_NAME = "Neko-Institute-of-Science/LLaMA-65B-4bit-128g"
    MODEL_BASE = "llama-65b-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, bits=4, group=128, actorder=True)

def download_koala_13b_v2():   
    MODEL_NAME = "TheBloke/koala-13B-GPTQ-4bit-128g"
    MODEL_BASE = "koala-13B-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+".safetensors"])
    save_meta(MODEL_NAME, MODEL_BASE, bits=4, group=128, actorder=True)

def download_wizardlm_1p0_30b_nogroup_model_v2():   
    MODEL_NAME = "TheBloke/WizardLM-30B-GPTQ"
    MODEL_BASE = "wizardlm-30b-GPTQ-4bit--1g.act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE)

def download_wizardlm_1p0_13b_v2():   
    MODEL_NAME = "TheBloke/wizardLM-13B-1.0-GPTQ"
    MODEL_BASE = "WizardLM-13B-1.0-GPTQ-4bit-128g.no-act-order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_vicuna_1p0_13b_safetensors_v2():   
    MODEL_NAME = "anon8231489123/vicuna-13b-GPTQ-4bit-128g"
    MODEL_BASE = "vicuna-13b-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE)

def download_vicuna_1p1_13b_safetensors_v2():   
    MODEL_NAME = "mzedp/vicuna-13b-v1.1-GPTQ-4bit-128g"
    MODEL_BASE = "vic-v1-13b-4b-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE)

def download_vicuna_1p3p0_13b_safetensors_v2():   
    MODEL_NAME = "TheBloke/vicuna-13b-v1.3.0-GPTQ"
    MODEL_BASE = "vicuna-13b-v1.3.0-GPTQ-4bit-128g.no-act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_vicuna_1p3_33b_safetensors_v2():   
    MODEL_NAME = "TheBloke/vicuna-33B-GPTQ"
    MODEL_BASE = "vicuna-33b-GPTQ-4bit--1g.act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=True, group=-1)

def download_vicuna_1p3_7b_safetensors_v2():   
    MODEL_NAME = "TheBloke/vicuna-7B-v1.3-GPTQ"
    MODEL_BASE = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)
              
def download_minotaur_15b_v2():   
    MODEL_NAME = "TheBloke/minotaur-15B-GPTQ"
    MODEL_BASE = "gptq_model-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_gpt4_alpaca_lora_65b_128g_v2():   
    MODEL_NAME = "TheBloke/gpt4-alpaca-lora_mlp-65B-GPTQ"
    MODEL_BASE = "gpt4-alpaca-lora_mlp-65B-GPTQ-4bit"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, bits=4, group=-1, actorder=True)

def download_airoboros_1p2_13b_v2():   
    MODEL_NAME = "TheBloke/airoboros-13B-gpt4-1.2-GPTQ"
    MODEL_BASE = "airoboros-13b-gpt4-1.2-GPTQ-4bit-128g.no-act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_airoboros_1p2_33b_v2():   
    MODEL_NAME = "TheBloke/airoboros-33B-gpt4-1.2-GPTQ"
    MODEL_BASE = "airoboros-33b-gpt4-1.2-GPTQ-4bit--1g.act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, group=-1)

def download_airoboros_1p4_13b_v2():   
    MODEL_NAME = "TheBloke/airoboros-13B-gpt4-1.4-GPTQ"
    MODEL_BASE = "airoboros-13b-gpt4-1.4-GPTQ-4bit-128g.no-act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_airoboros_1p4_33b_v2():   
    MODEL_NAME = "TheBloke/airoboros-33B-gpt4-1.4-GPTQ"
    MODEL_BASE = "airoboros-33b-gpt4-1.4-GPTQ-4bit--1g.act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=True, group=-1)

def download_airoboros_1p4_65b_v2():   
    MODEL_NAME = "TheBloke/airoboros-65B-gpt4-1.4-GPTQ"
    MODEL_BASE = "airoboros-65b-gpt4-1.4-GPTQ-4bit--1g.act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=True, group=-1)
    
def download_robin_13b_v2():   
    MODEL_NAME = "TheBloke/robin-13B-v2-GPTQ"
    MODEL_BASE = "robin-13b-GPTQ-4bit-128g.no-act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_ultralm_13b_v2():   
    MODEL_NAME = "TheBloke/UltraLM-13B-GPTQ"
    MODEL_BASE = "ultralm-13b-GPTQ-4bit-128g.no-act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_orcamini2_7b_v2():   
    MODEL_NAME = "TheBloke/orca_mini_v2_7B-GPTQ"
    MODEL_BASE = "orca-mini-v2_7b-GPTQ-4bit-128g.no-act.order"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

def download_llama2_13b_v2():   
    MODEL_NAME = "TheBloke/Llama-2-13B-GPTQ"
    MODEL_BASE = "gptq_model-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model","*.txt","*.py",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE)


def download_llama2_70b_v2():   
    MODEL_NAME = "Panchovix/LLaMA-2-70B-GPTQ-transformers4.32.0.dev0"
    MODEL_BASE = "llama-2-70b-4bit"

    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model","*.txt","*.py",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, group=-1, actorder=True)

def download_upstage_instruct_2048_v2():
    MODEL_NAME = "TheBloke/upstage-llama-30b-instruct-2048-GPTQ"
    MODEL_BASE = "gptq_model-4bit--1g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model","*.txt","*.py",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, group=-1, actorder=True)

def download_codeCherryPop_7b_v2():   
    MODEL_NAME = "TheBloke/llama2-7b-chat-codeCherryPop-qLoRA-GPTQ"
    MODEL_BASE = "gptq_model-4bit-128g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, actorder=False)

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
        "git clone https://github.com/turboderp/exllama /repositories/exllama && cd /repositories/exllama && git checkout cade9bc5576292056728cf55c0c9faf4adae62f8",
        "cd /repositories/exllama && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "cd /repositories/exllama && pip install safetensors sentencepiece ninja huggingface_hub hf_transfer",
        gpu="any",
    )
    #### SELECT MODEL HERE ####
    .run_function(download_codeCherryPop_7b_v2)
)

### SELECT count=1 A10G (up to 30B) or count=2 A10G (for 65B)
gpu_request = gpu.A10G(count=1)
gpu_split = '17,24' if gpu_request.count == 2 else None

## Entrypoint import trick for when inside the remote container
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
@stub.cls(image=stub.gptq_image, gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300)
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
        if gpu_split is not None:
            self.config.set_auto_map(gpu_split)

        print('Loading model...')
        self.model = ExLlama(self.config)
        self.cache = ExLlamaCache(self.model)
        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.tokenizer = ExLlamaTokenizer(tokenizer_model_path)

    def params(self, temperature=0.7, repetition_penalty=1.0, repeat_last_n=256, repetition_decay=128, top_k=-1, top_p=1.0, max_new_tokens=512, beams=1, beam_length=1, **kwargs):
        return {
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "repeat_last_n": repeat_last_n,
            "repetition_decay": repetition_decay,
            "top_k": top_k if top_k > 0 else 1000,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "beams": beams,
            "beam_length": beam_length,
            "stop_seq": "###"
        }

    @method()
    def generate(self, prompt, params):
        # Init generator
        generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        generator.settings = ExLlamaGenerator.Settings()
        generator.settings.temperature = params['temperature']
        generator.settings.top_k = params['top_k']
        generator.settings.top_p = params['top_p']
        generator.settings.min_p = 0
        generator.settings.token_repetition_penalty_max = params['repetition_penalty']
        generator.settings.token_repetition_penalty_sustain = params['repeat_last_n']
        generator.settings.token_repetition_penalty_decay = params['repetition_decay']

        # Beam Search Parameters
        generator.settings.beams = params['beams']
        generator.settings.beam_length = params['beam_length']

        # Encode prompt and init the generator
        prompt_ids = self.tokenizer.encode(prompt)
        num_res_tokens = prompt_ids.shape[-1]
        generator.gen_begin(prompt_ids)

        # Begin beam search
        generator.begin_beam_search()
        min_response_tokens = 10
        res_line = ''

        t0 = time.time()

        for i in range(params['max_new_tokens']):

            # Disallowing the end condition tokens seems like a clean way to force longer replies.
            if i < min_response_tokens:
                generator.disallow_tokens([self.tokenizer.eos_token_id])
            else:
                generator.disallow_tokens(None)

            # Get a token
            gen_token = generator.beam_search()

            # If token is EOS, replace it with newline before continuing
            if gen_token.item() == self.tokenizer.eos_token_id:
                generator.replace_last_token(self.tokenizer.newline_token_id)

            # Decode the current line and print any characters added
            num_res_tokens += 1
            text = self.tokenizer.decode(generator.sequence_actual[:, -num_res_tokens:][0])
            new_text = text[len(res_line):]
            res_line += new_text    

            print(new_text, end="")  # (character streaming output is here)
            sys.stdout.flush()

            # End conditions
            if gen_token.item() == self.tokenizer.eos_token_id: break
            if res_line.endswith(params['stop_seq']): break

            generator.end_beam_search()
            answer = text[len(prompt)+1:]

        print(f"Generated {num_res_tokens-prompt_ids.shape[-1]} tokens in {time.time()-t0:.2f}s")
        return answer, self.info

# For local testing, run `modal run -q interview-exllama-modal.py --input prepare_junior-dev_python.ndjson --params params/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str, iterations: int = 1):
    from prepare import save_interview

    model = ModalExLlama()
    model_info = None

    tasks = []
    for param_file in params.split(','):
        for input_file in input.split(','):
            tasks.append((param_file, input_file))

    for param_file, input_file in tasks:
      interview = [json.loads(line) for line in open(input_file)]
      params_json = json.load(open(param_file,'r'))
      params_model = model.params(**params_json)

      print('Executing with parameter file',param_file,'and input file',input_file)

      for iter in range(iterations):
        results = []
        for question in interview:
            print(iter, param_file, question['name'], question['language'])

            # generate the answer
            t0 = time.time()
            answer, info = model.generate.call(question['prompt'], params=params_model)
            elapsed = time.time() - t0

            # save for later
            if model_info is None:
                model_info = info
                print('Local model info:', model_info)
            
            print()
            print(answer)
            print(f"Generated in {elapsed:.2f}s")

            result = question.copy()
            result['answer'] = answer
            result['params'] = params_model
            result['model'] = info['model_name']
            result['runtime'] = 'exllama'
            results.append(result)

        save_interview(input_file, 'none', param_file, model_info['model_name'], results)