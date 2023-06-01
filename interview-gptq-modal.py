import time
import json
import pandas as pd
import os
from pathlib import Path
from modal import Image, Stub, method, create_package_mounts, gpu
from jinja2 import Template

#### SEE NOTE BELOW! ####
MODEL_NAME = "TheBloke/VicUnlocked-30B-LoRA-GPTQ"
MODEL_FILES = ["*"]
MODEL_WBITS = 4
MODEL_GROUPSIZE = -1 # -1 to disable

stub = Stub(name=MODEL_NAME.replace('/', '-'))

#### NOTE: Modal will not rebuild the container unless this function name or it's code contents change.
####       It is NOT sufficient to change any of the constants above.
####       Suggestion is to rename this function after the model.
def download_VicUnlocked_model():
    from huggingface_hub import snapshot_download

    # Match what FastChat expects
    # https://github.com/thisserand/FastChat/blob/4a57c928a906705404eae06f7a44b4da45828487/download-model.py#L203
    output_folder = f"{'_'.join(MODEL_NAME.split('/')[-2:])}"

    snapshot_download(
        local_dir=Path("/FastChat", "models", MODEL_NAME.replace('/', '_')),
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
        "git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda /FastChat/repositories/GPTQ-for-LLaMa",
        "cd /FastChat/repositories/GPTQ-for-LLaMa && pip install -r requirements.txt && python setup_cuda.py install",
        gpu="any",
    )
    .run_function(download_VicUnlocked_model)
)


#### NOTE: If you've renamed the download function, also change the call above!

if stub.is_inside(stub.gptq_image):
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    import sys
    sys.path.insert(0, str(Path("/FastChat/repositories/GPTQ-for-LLaMa")))
    from pathlib import Path
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    from modelutils import find_layers
    from quant import make_quant

    # https://github.com/thisserand/FastChat/blob/main/fastchat/serve/load_gptq_model.py
    def load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=True, exclude_layers=['lm_head'], kernel_switch_threshold=128):
        config = AutoConfig.from_pretrained(model)
        def noop(*args, **kwargs):
            pass
        torch.nn.init.kaiming_uniform_ = noop 
        torch.nn.init.uniform_ = noop 
        torch.nn.init.normal_ = noop 

        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = AutoModelForCausalLM.from_config(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in exclude_layers:
            if name in layers:
                del layers[name]
        make_quant(model, layers, wbits, groupsize, faster=faster_kernel, kernel_switch_threshold=kernel_switch_threshold)

        del layers
        
        print('Loading model ...')
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            model.load_state_dict(safe_load(checkpoint))
        else:
            model.load_state_dict(torch.load(checkpoint))
        model.seqlen = 2048
        print('Done.')

        return model

    # https://github.com/thisserand/FastChat/blob/main/fastchat/serve/load_gptq_model.py
    def load_quantized(model_name, wbits=4, groupsize=128, threshold=128):
        model_name = model_name.replace('/', '_')
        path_to_model = Path(f'/FastChat/models/{model_name}')
        found_pts = list(path_to_model.glob("*.pt"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        pt_path = None

        if len(found_pts) == 1:
            pt_path = found_pts[0]
        elif len(found_safetensors) == 1:
            pt_path = found_safetensors[0]

        if not pt_path:
            print("Could not find the quantized model in .pt or .safetensors format, exiting...")
            exit()

        model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, kernel_switch_threshold=threshold)

        return model

    #https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py
    def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
    ) -> LogitsProcessorList:   
        processor_list = LogitsProcessorList()
        # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list

    # https://github.com/thisserand/FastChat/blob/main/fastchat/serve/cli.py
    # partially merged with https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py
    @torch.inference_mode()
    def generate_stream(tokenizer, model, params, device,
                        context_len=2048, stream_interval=2):
        """Adapted from fastchat/serve/model_worker.py::generate_stream"""

        prompt = params["prompt"]
        l_prompt = len(prompt)
        
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable

        logits_processor = prepare_logits_processor(temperature, repetition_penalty, top_p, top_k)

        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=device)
                out = model(input_ids=torch.as_tensor([[token]], device=device),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break

        del past_key_values

#### NOTE: SET GPU TYPE HERE ####
@stub.cls(image=stub.gptq_image, gpu=gpu.A10G(count=1), concurrency_limit=1, container_idle_timeout=300)
class ModalGPTQ:
    def __enter__(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Loading GPTQ quantized model...")
        model = load_quantized(MODEL_NAME, wbits=MODEL_WBITS, groupsize=MODEL_GROUPSIZE)
        model.cuda()

        self.model = model
        self.tokenizer = tokenizer
        print(f"Model loaded in {time.time() - t0:.2f}s")

    def params(self, temperature=0.7, repetition_penalty=1.0, top_k=-1, top_p=1.0, max_new_tokens=512, stop='###', **kwargs):
        return {
            "model": MODEL_NAME,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "top_p": top_p,
            "stop": stop,
            "max_new_tokens": max_new_tokens
        }

    @method()
    async def generate(self, input, params=None):
        if input == "":
            raise Exception("Input is empty")

        if params is None:
            params = self.params()

        params['prompt'] = input
        print(params)

        prev = len(input) + 1
        count = 0
        t0 = time.time()
        for outputs in generate_stream(self.tokenizer, self.model, params, "cuda"):
            yield outputs[prev:]
            prev = len(outputs)
            count = count + 2 # stream_interval
        dur = time.time() - t0

        print(f"{count} tokens generated in {dur:.2f}s, {1000*dur/count:.2f} ms/token", file=sys.stderr)

# For local testing, run `modal run -q interview-gptq-modal.py --input questions.csv --params model_parameters/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str):

    model = ModalGPTQ()

    interview = [json.loads(line) for line in open(input)]
    params_json = json.load(open(params,'r'))
    params_model = model.params(**params_json)

    results = []
    for question in interview:
        print(question['name'], question['language'])

        answer = ""
        for val in model.generate.call(question['prompt'], params=params_model):
            answer += val
            print(val, end="", flush=True)

        print()

        result = question.copy()
        result['answer'] = answer
        result['params'] = params_model
        result['model'] = params_model['model']
        results.append(result)

    # Save results
    [stage, interview_name, languages, template, *stuff] = Path(args.input).stem.split('_')
    templateout_name = 'none'
    params_name = Path(params).stem
    model_name = results[0]['model'].replace('/','-')
    ts = str(int(time.time()))

    output_filename = 'results/'+'_'.join(['interview', interview_name, languages, template, templateout_name, params_name, model_name, ts])+'.ndjson'
    with open(output_filename, 'w') as f:
        f.write('\n'.join([json.dumps(result, default=vars) for result in results]))
    print('Saved results to', output_filename)