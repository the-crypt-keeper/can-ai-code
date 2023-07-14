import time
import json
from pathlib import Path
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
import sys
#sys.path.insert(0, str(Path("/repositories/AutoGPTQ")))
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.modeling import BaseQuantizeConfig
import fire

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

def download_falcon_40b_3bit_v2():   
    MODEL_NAME = "TheBloke/falcon-40b-instruct-3bit-GPTQ"
    MODEL_BASE = "gptq_model-3bit--1g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model","*.txt","*.py",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, group=-1, bits=3, actorder=True, eos=['<|endoftext|>'])

def download_falcon_40b_4bit_v2():   
    MODEL_NAME = "TheBloke/falcon-40b-instruct-GPTQ"
    MODEL_BASE = "gptq_model-4bit--1g"

    snapshot_download(local_dir=Path("/model"), repo_id=MODEL_NAME, allow_patterns=["*.json","*.model","*.txt","*.py",MODEL_BASE+"*"])
    save_meta(MODEL_NAME, MODEL_BASE, group=-1, bits=4, actorder=True, eos=['<|endoftext|>'])


class ModalGPTQ:
    def __enter__(self):
        quantized_model_dir = "/model"

        self.info = json.load(open('/model/_info.json'))
        print('Remote model info:', self.info)

        if not Path('/model/quantize_config.json').exists():
            quantize_config = BaseQuantizeConfig()
            quantize_config.desc_act = self.info['model_actorder']
            quantize_config.bits = self.info['model_bits']
            quantize_config.group = self.info['model_group']
            quantize_config.save_pretrained('/model')
        else:
            print('This model contains quantize_config.')

        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False)

        print('Loading model...')
        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, model_basename=self.info['model_base'], device_map="auto", load_in_8bit=True, use_triton=False, use_safetensors=self.info['model_safetensors'], torch_dtype=torch.float32, trust_remote_code=True)
        
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

    def generate(self, prompt, params):
        tokens = self.tokenizer(prompt, return_tensors="pt").to("cuda:0").input_ids
        output = self.model.generate(input_ids=tokens, do_sample=True, **params)

        decoded = self.tokenizer.decode(output[0])

        # Remove the prompt and all special tokens
        answer = decoded.replace(prompt, '')
        for special_token in self.info['model_eos']:
            answer = answer.replace(special_token, '')

        return answer, self.info

def main(input: str, params: str, iterations: int = 1):
    from prepare import save_interview

    download_falcon_40b_4bit_v2()
    model = ModalGPTQ()

    interview = [json.loads(line) for line in open(input)]
    params_json = json.load(open(params,'r'))
    params_model = model.params(**params_json)
    model_info = None

    for iter in range(iterations):
        results = []
        for idx, question in enumerate(interview):
            print(f"[{idx+1}/{len(interview)}] {question['language']} {question['name']}")

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
            result['runtime'] = 'autogptq'
            results.append(result)

        save_interview(input, 'none', params, model_info['model_name'], results)

if __name__ == "__main__":
    fire.Fire(main)