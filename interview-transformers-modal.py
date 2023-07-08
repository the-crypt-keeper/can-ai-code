from modal import Stub, Image, method, gpu
from huggingface_hub import snapshot_download
import time
import json
from jinja2 import Template

def download_model(name):
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name}, f)
    snapshot_download(name)
    snapshot_download("hf-internal-testing/llama-tokenizer")

def download_codegen_2p5_7b_mono_model():
    download_model("Salesforce/codegen25-7b-mono")

# Now, we define our image. We’ll start from a Dockerhub image recommended by `vLLM`, upgrade the older
# version of `torch` to a new one specifically built for CUDA 11.8. Next, we install `vLLM` from source to get the latest updates.
# Finally, we’ll use run_function to run the function defined above to ensure the weights of the model
# are saved within the container image.
image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.30.2",
        "tiktoken==0.4.0"
    )
    .run_function(download_codegen_2p5_7b_mono_model)
)

stub = Stub(image=image)

gpu_request = gpu.A100(count=1)
@stub.cls(gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300)
class ModalTransformers:
    def __enter__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.info = json.load(open('./_info.json'))
        print('Remote model info:', self.info)

        t0 = time.time()
        print('Starting up...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.info['model_name'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.info['model_name'], trust_remote_code=True)
        print(f"Model loaded in {time.time() - t0:.2f}s")        
           
    @method()
    def generate(self, prompt, params):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        sampling_params = {
            'temperature': params['temperature'],
            'max_length': params['max_new_tokens'],
            'top_k': params['top_k'],
            'top_p': params['top_p'],
            'repetition_penalty': params['repetition_penalty']
        }
        sample = self.model.generate(input_ids, max_length=128) #generate(**inputs, do_sample=True, **sampling_params)
        self.info['sampling_params'] = sampling_params
        answer = self.tokenizer.decode(sample[0])
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
        for question in interview:
            print(question['name'], question['language'])

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

        save_interview(input, 'none', params, model_info['model_name'], results)
