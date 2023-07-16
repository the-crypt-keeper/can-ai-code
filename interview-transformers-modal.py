from modal import Stub, Image, method, gpu
from huggingface_hub import snapshot_download
import time
import json
from jinja2 import Template

def download_model(name, **kwargs):
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name}, f)
    snapshot_download(name, **kwargs)
    snapshot_download("hf-internal-testing/llama-tokenizer")

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

# Now, we define our image. We’ll start from a Dockerhub image recommended by `vLLM`, upgrade the older
# version of `torch` to a new one specifically built for CUDA 11.8. Next, we install `vLLM` from source to get the latest updates.
# Finally, we’ll use run_function to run the function defined above to ensure the weights of the model
# are saved within the container image.
image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:23.06-py3")
    .pip_install(
        "transformers==4.30.2",
        "tiktoken==0.4.0",
        "bitsandbytes==0.40.1.post1",
        "accelerate==0.21.0"
    )
    .pip_install("einops==0.6.1", "sentencepiece==0.1.99")
    .run_function(download_vicuna_1p3_7b_model)
)

stub = Stub(image=image)

gpu_request = gpu.A10G(count=1)
@stub.cls(gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300)
class ModalTransformers:
    def __enter__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        self.info = json.load(open('./_info.json'))
        print('Remote model info:', self.info)

        # Select FP32 or FP16 here
        torch_dtype = torch.float16
        # Enable quants here
        quantization_config = BitsAndBytesConfig(load_in_8bit = False,
                                                 load_in_4bit = False,
                                                 bnb_4bit_quant_type = "fp4")

        t0 = time.time()
        print('Starting up...', str(torch_dtype))
        self.tokenizer = AutoTokenizer.from_pretrained(self.info['model_name'], trust_remote_code=True)
        print('Loading model...')
        self.model = AutoModelForCausalLM.from_pretrained(self.info['model_name'], device_map="auto", torch_dtype=torch_dtype, quantization_config=quantization_config, trust_remote_code=True)
        print(f"Model loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")

        if quantization_config.load_in_4bit:
            print('Loaded in fp4.')
            self.info['model_name'] = self.info['model_name'] + '-' + quantization_config.bnb_4bit_quant_type
        elif quantization_config.load_in_8bit:
            print('Loaded in int8.')
            self.info['model_name'] = self.info['model_name'] + '-int8'
        elif torch_dtype == torch.float16:
            print('Loaded in fp16.')
            self.info['model_name'] = self.info['model_name'] + '-fp16'

    @method()
    def generate(self, prompt, params):
        input = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input.input_ids.to('cuda')
        attention_mask = input.attention_mask.to('cuda')

        sampling_params = {
            'do_sample': True,
            'temperature': params.get('temperature', 1.0),
            'max_length': params.get('max_new_tokens', 512),
            'top_k': params.get('top_k', 40),
            'top_p': params.get('top_p', 1.0),
            'repetition_penalty': params.get('repetition_penalty', 1.0)
        }
        sample = self.model.generate(input_ids, attention_mask=attention_mask, eos_token_id=self.tokenizer.eos_token_id, **sampling_params)
        self.info['sampling_params'] = sampling_params
        answer = self.tokenizer.decode(sample[0], skip_special_tokens=True)[len(prompt):]
        return answer, self.info
    
# For local testing, run `modal run -q interview-transformers-modal.py --input questions.csv --params model_parameters/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str, iterations: int = 1, templateout: str = ""):
    from prepare import save_interview

    model = ModalTransformers()

    interview = [json.loads(line) for line in open(input)]
    params_json = json.load(open(params,'r'))
    model_info = None

    output_template = Template(open(templateout).read()) if templateout else None

    for iter in range(iterations):
        results = []
        for idx, question in enumerate(interview):
            print(f"{idx+1}/{len(interview)} {question['name']} {question['language']}")

            # generate the answer
            result, info = model.generate.call(question['prompt'], params=params_json)

            # save for later
            if model_info is None:
                model_info = info
                print('Local model info:', model_info)

            # optional output template
            answer = output_template.render(**question, Answer=result) if output_template else result
            
            print()
            print(answer)
            print()

            result = question.copy()
            result['answer'] = answer
            result['params'] = info['sampling_params']
            result['model'] = info['model_name']
            result['runtime'] = 'transformers'
            results.append(result)

        save_interview(input, templateout if templateout else 'none', params, model_info['model_name'], results)
