from modal import Stub, Image, method, gpu
from huggingface_hub import snapshot_download
import time
import json

def download_model(name):
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name}, f)
    snapshot_download(name)
    snapshot_download("hf-internal-testing/llama-tokenizer")

# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# Since `vLLM` uses the default Huggingface cache location, we can use library functions to pre-download the model into our image.
def download_vicuna_7b_1p3_model():
    download_model("lmsys/vicuna-7b-v1.3")

def download_vicuna_13b_1p3_model():
    download_model("lmsys/vicuna-13b-v1.3")

def download_airoboros_7b_1p4_model():
    download_model("jondurbin/airoboros-7b-gpt4-1.4-fp16")

def download_airoboros_7b_1p4p1_model():
    download_model("jondurbin/airoboros-7b-gpt4-1.4.1-qlora")

def download_airoboros_13b_1p4_model():
    download_model("jondurbin/airoboros-13b-gpt4-1.4-fp16")

def download_airoboros_13b_1p4p1_model():
    download_model("jondurbin/airoboros-13b-gpt4-1.4.1-qlora")

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
        "vllm @ git+https://github.com/vllm-project/vllm.git@2b7d3aca2e1dd25fe26424f57c051af3b823cd71"
    )
    .run_function(download_airoboros_13b_1p4_model)
)

stub = Stub(image=image)

gpu_request = gpu.A100(count=1)
@stub.cls(gpu=gpu_request, concurrency_limit=1, container_idle_timeout=300)
class ModalVLLM:
    def __enter__(self):
        from vllm import LLM

        self.info = json.load(open('./_info.json'))
        print('Remote model info:', self.info)

        t0 = time.time()
        print('Starting up...')
        self.llm = LLM(model=self.info['model_name'])  # Load the model
        print(f"Model loaded in {time.time() - t0:.2f}s")        
        
   
    @method()
    def generate(self, prompt, params):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=params['temperature'], top_k=params['top_k'], top_p=params['top_p'], max_tokens=params['max_new_tokens'], presence_penalty=params['repetition_penalty']
        )
        result = self.llm.generate(prompt, sampling_params)
        self.info['sampling_params'] = str(sampling_params)

        answers = []
        for i in range(len(prompt)):
            for r in result:
                if r.prompt == prompt[i]:
                    answers.append(r.outputs[0].text.replace('</s>','').replace('<|endoftext|>',''))
                    break

        return answers, self.info

# For local testing, run `modal run -q interview-vllm-modal.py --input prepare_junior-dev_python.ndjson --params params/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str, iterations: int = 1):
    from prepare import save_interview

    model = ModalVLLM()

    tasks = []
    for param_file in params.split(','):
        for input_file in input.split(','):
            tasks.append((param_file, input_file))

    for param_file, input_file in tasks:
      interview = [json.loads(line) for line in open(input_file)]
      params_json = json.load(open(param_file,'r'))

      print('Executing with parameter file',param_file,'and input file',input_file)

      for iter in range(iterations):
        print(iter, param_file, len(interview))

        # generate the answers
        t0 = time.time()
        prompts = [q['prompt'] for q in interview]
        answers, info = model.generate.call(prompts, params=params_json)
        elapsed = time.time() - t0
        print(f"Generated in {elapsed:.2f}s")

        # save results
        results = [q.copy() for q in interview]

        for i, answer in enumerate(answers):
            results[i]['answer'] = answer
            results[i]['params'] = info['sampling_params']
            results[i]['model'] = info['model_name']
            results[i]['runtime'] = 'vllm'

            print(prompts[i])
            print(answer)
            print()

        save_interview(input_file, 'none', param_file, info['model_name'], results)