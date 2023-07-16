from modal import Stub, Image, method, gpu
from huggingface_hub import snapshot_download
import time
import json
from jinja2 import Template

def download_awq_model(name, base_model, q_group_size=64, w_bit=4, big_model=False):
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name, "base_model": base_model, "q_group_size": q_group_size, "w_bit": w_bit, "big_model": big_model}, f)
    snapshot_download(name)
    snapshot_download(base_model, allow_patterns=["*.json","*.model"])

def download_awq_model_v2(name, q_group_size=64, w_bit=4, big_model=False):
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name, "base_model": None, "q_group_size": q_group_size, "w_bit": w_bit, "big_model": big_model}, f)
    snapshot_download(name)

def download_awq_falcon_instruct_7b_model():
    # pre-quantized model required
    download_awq_model("abhinavkulkarni/falcon-7b-instruct-w4-g64-awq", "tiiuae/falcon-7b-instruct")

def download_awq_codgen2p5_7b_model():
    download_awq_model("abhinavkulkarni/Salesforce-codegen25-7b-multi-w4-g128-awq", "Salesforce/codegen25-7b-multi", q_group_size=128)

def download_awq_falcon_instruct_40b_model():
    download_awq_model("abhinavkulkarni/tiiuae-falcon-40b-instruct-w4-g128-awq", "tiiuae/falcon-40b-instruct", q_group_size=128, big_model=True)

def download_awq_vicuna_1p3_33b_model():
    download_awq_model_v2("abhinavkulkarni/lmsys-vicuna-33b-v1.3-w4-g128-awq", q_group_size=128)  

image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:23.06-py3")
    .pip_install(
        "transformers==4.30.2",
        "tiktoken==0.4.0",
        "bitsandbytes==0.39.1",
        "accelerate==0.19.0"
    )
    .run_commands("git clone https://github.com/mit-han-lab/llm-awq",
                  "cd llm-awq && git checkout 71d8e68df78de6c0c817b029a568c064bf22132d && pip install -e .")
    .run_commands("cd llm-awq/awq/kernels && export TORCH_CUDA_ARCH_LIST='8.0 8.6 8.7 8.9 9.0' && python setup.py install")
    .run_function(download_awq_vicuna_1p3_33b_model)
)

stub = Stub(image=image)

gpu_request = gpu.A10G(count=1)
@stub.cls(gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300)
class ModalTransformers:
    def __enter__(self):
        import torch
        from awq.quantize.quantizer import real_quantize_model_weight
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
        from huggingface_hub import snapshot_download

        self.info = json.load(open('./_info.json'))
        print('Remote model info:', self.info)

        # Config
        base_model = self.info['base_model'] if self.info['base_model'] else self.info['model_name']
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        # Model
        t0 = time.time()
        print('Starting up...')

        load_quant = snapshot_download(self.info['model_name'])

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)

        q_config = { "zero_point": True, "q_group_size": self.info['q_group_size'] }
        real_quantize_model_weight(model, w_bit=self.info['w_bit'], q_config=q_config, init_only=True)

        if self.info['big_model']:
            print('Loading big model with gpu_count', gpu_request.count)
            max_memory = {0:"18GiB", "cpu":"99GiB"} if gpu_request.count == 1 else { 0:"18GiB", 1:"22GiB" }
            device_map = infer_auto_device_map(model,
                                               no_split_module_classes=["DecoderLayer"],
                                               max_memory=max_memory)
            if device_map['lm_head'] == 'cpu': device_map['lm_head'] = 0
            print(device_map)
        else:
            device_map = 'balanced'
        
        self.model = load_checkpoint_and_dispatch(model, load_quant, device_map=device_map)

        print(f"Model loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")

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
        sample = self.model.generate(input_ids, attention_mask=attention_mask, use_cache=True, eos_token_id=self.tokenizer.eos_token_id, **sampling_params)
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
            print(f"[{idx+1}/{len(interview)}] {question['language']} {question['name']}")

            # generate the answer
            start_time = time.time()
            result, info = model.generate.call(question['prompt'], params=params_json)
            print(f"Answer generated in {time.time() - start_time:.2f}s")

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
            result['runtime'] = 'awq'
            results.append(result)

        save_interview(input, templateout if templateout else 'none', params, model_info['model_name'], results)
