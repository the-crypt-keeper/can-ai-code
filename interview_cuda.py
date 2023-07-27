#!/usr/bin/env python3
import time
import json
from jinja2 import Template

#########################################
##  Transformers/BitsAndBytes Adapter  ##
#########################################

QUANT_FP32 = 0
QUANT_FP16 = 1
QUANT_INT8 = 10
QUANT_FP4  = 20

quant_suffix = {}
quant_suffix[QUANT_FP32] = 'fp32'
quant_suffix[QUANT_FP16] = 'fp16'
quant_suffix[QUANT_INT8] = 'int8'
quant_suffix[QUANT_FP4] = 'fp4'

class InterviewTransformers:
    def __init__(self, model_name, model_info = {}, quant = QUANT_FP16, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        print('Remote model', self.model_name, ' info', self.info)

        t0 = time.time()
        print('Loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, **self.info.get('tokenizer_args', {}))

        print('Loading model with accelerate...')
        torch_dtype = torch.float32 if self.quant == QUANT_FP32 else torch.float16
        quantization_config = BitsAndBytesConfig(load_in_8bit = self.quant == QUANT_INT8,
                                                load_in_4bit = self.quant == QUANT_FP4,
                                                bnb_4bit_quant_type = "fp4")            
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch_dtype, quantization_config=quantization_config, trust_remote_code=True)
        
        # if passed a path, take the last dir name otherwise replace / with -
        if self.model_name[0] == '/':
            self.info['model_name'] = self.model_name.split('/')[-1]
        else:
            self.info['model_name'] = self.model_name.replace('/','-')
        # add quant suffix
        if self.quant in quant_suffix:
            self.info['model_name'] += '-' + quant_suffix[self.quant]

        print(f"Model {self.info['model_name']} loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")        

    def generate(self, prompt, params):
        from transformers import GenerationConfig

        generation_config, unused_kwargs = GenerationConfig.from_pretrained(
            self.model_name, do_sample = True, **params, return_unused_kwargs=True
        )
        self.info['sampling_params'] = str(generation_config)

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        sample = self.model.generate(inputs, generation_config=generation_config)
        answer = self.tokenizer.decode(sample[0]).replace(prompt, '').replace('<|endoftext|>','').replace('</s>','').replace('<s>','')
        return answer, self.info

#########################
##  auto-gptq Adapter  ##
#########################
class InterviewAutoGPTQ:
    def __init__(self, model_name, model_info = {}, quant = None, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = False

    def load(self):
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
        import torch

        print('Remote model', self.model_name, ' info', self.info)

        t0 = time.time()
        print('Loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, **self.info.get('tokenizer_args', {}))

        # TODO: support models without quantize config specified
        #quantize_config = BaseQuantizeConfig()
        #quantize_config.desc_act = self.info['model_actorder']
        #quantize_config.bits = self.info['model_bits']
        #quantize_config.group = self.info['model_group']

        print('Loading model with autogptq...')        
        self.model = AutoGPTQForCausalLM.from_quantized(self.model_name, device_map="auto", use_triton=False, use_safetensors=self.info.get('model_safetensors', True), torch_dtype=torch.float32, trust_remote_code=True)
    
        # if passed a path, take the last dir name otherwise replace / with -
        if self.model_name[0] == '/':
            self.info['model_name'] = self.model_name.split('/')[-1]
        else:
            self.info['model_name'] = self.model_name.replace('/','-')

        print(f"Model {self.info['model_name']} loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")        

    def generate(self, prompt, params):
        sampling_params = {
            "temperature": params.get('temperature', 1.0),
            "repetition_penalty": params.get('repetition_penalty', 1.0),
            "top_k": params.get('top_k', 1000),
            "top_p": params.get('top_p', 1.0),
            "max_new_tokens": params.get('max_new_tokens', 512)
        }
        self.info['sampling_params'] = sampling_params

        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        sample = self.model.generate(input_ids=tokens, do_sample=True, **sampling_params)
        answer = self.tokenizer.decode(sample[0]).replace(prompt, '').replace('<|endoftext|>','').replace('</s>','').replace('<s>','')
        return answer, self.info

#######################
##  exllama Adapter  ##
#######################
class InterviewExllama:
    def __init__(self, model_name, model_info = {}, quant = None, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.gpu_split = gpu_split
        self.quant = quant
        self.batch = False

    def load(self):
        import sys
        sys.path.insert(0, "/repositories/exllama")
        print('Starting up...')
        import torch
        from model import ExLlama, ExLlamaCache, ExLlamaConfig
        from tokenizer import ExLlamaTokenizer
        from huggingface_hub import hf_hub_download, HfApi

        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()

        print("Loading tokenizer..")
        tokenizer_model_path = hf_hub_download(repo_id=self.model_name, filename="tokenizer.model")
        self.tokenizer = ExLlamaTokenizer(tokenizer_model_path)

        api = HfApi()
        files = api.list_files_info(self.model_name)
        model_path = None
        for file_info in files:
            if (file_info.rfilename.find(".safetensors") != -1):
                model_path = hf_hub_download(repo_id=self.model_name, filename=file_info.rfilename)
                break
        
        if model_path is None:
            raise Exception("Could not find safetensors.")
        else:
            print("Loading from", model_path)
        
        self.config = ExLlamaConfig(hf_hub_download(repo_id=self.model_name, filename="config.json"))
        self.config.model_path = model_path
        self.config.max_seq_len = self.info.get('max_seq_len', 2048)
        if self.gpu_split is not None:
            self.config.set_auto_map(self.gpu_split)

        print('Loading model...')
        t0 = time.time()
        self.model = ExLlama(self.config)
        self.cache = ExLlamaCache(self.model)
        print(f"Model loaded in {time.time() - t0:.2f}s")

    def generate(self, prompt, params):
        # Init generator
        from generator import ExLlamaGenerator
        import sys

        generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        generator.settings = ExLlamaGenerator.Settings()
        generator.settings.temperature = params.get('temperature', 1.0)
        generator.settings.top_k = params.get('top_k', 1000)
        generator.settings.top_p = params.get('top_p', 1.0)
        generator.settings.min_p = 0
        generator.settings.token_repetition_penalty_max = params.get('repetition_penalty', 1.0)
        generator.settings.token_repetition_penalty_sustain = params.get('repeat_last_n', 256)
        generator.settings.token_repetition_penalty_decay = params.get('repetition_decay', 128)

        # Beam Search Parameters
        generator.settings.beams = params.get('beams', 1)
        generator.settings.beam_length = params.get('beam_length', 1)

        self.info['sampling_params'] = str(generator.settings.__dict__)

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

            #print(new_text, end="")  # (character streaming output is here)
            #sys.stdout.flush()

            # End conditions
            if gen_token.item() == self.tokenizer.eos_token_id: break
            if res_line.endswith(params.get('stop_seq','###')): break

            generator.end_beam_search()
            answer = text[len(prompt)+1:]

        print(f"Generated {num_res_tokens-prompt_ids.shape[-1]} tokens in {time.time()-t0:.2f}s")
        return answer, self.info

####################
##  vLLM Adapter  ##
####################
class InterviewVLLM:
    def __init__(self, model_name, model_info = {}, quant = None, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = True
        self.gpu_split = gpu_split

    def load(self):
        from vllm import LLM

        print('Remote model', self.model_name, ' info', self.info)

        t0 = time.time()
        if self.gpu_split is not None:
            print('Starting in multi-gpu mode...')
            self.llm = LLM(model=self.model_name, tensor_parallel_size=self.gpu_split)
        else:
            print('Starting in single GPU mode..')
            self.llm = LLM(model=self.model_name)

        print(f"Model loaded in {time.time() - t0:.2f}s")   

    def generate(self, prompt, params):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=params.get('temperature', 1.0),
            top_k=params.get('top_k', 1000),
            top_p=params.get('top_p', 1.0),
            max_tokens=params.get('max_new_tokens', 512),
            presence_penalty=params.get('repetition_penalty', 1.0)
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
    
###################
##  AWQ Adapter  ##
###################
class InterviewAWQ:
    def __init__(self, model_name, model_info = {}, quant = None, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant
        self.gpu_split = gpu_split

        self.batch = True

    def load(self):
        import torch
        from awq.quantize.quantizer import real_quantize_model_weight
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
        from huggingface_hub import hf_hub_download, HfApi

        # Config
        print('Starting up...')
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Model
        t0 = time.time()

        api = HfApi()
        files = api.list_files_info(self.model_name)
        model_path = None
        search_list = [".index.json", ".pt", ".bin"]
        for file_info in files:
            for needle in search_list:
                if file_info.rfilename.find(needle) != -1:
                    model_path = hf_hub_download(repo_id=self.model_name, filename=file_info.rfilename)
                    break

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)

        q_config = { "zero_point": True, "q_group_size": self.info.get('q_group_size', 128) }
        real_quantize_model_weight(model, w_bit=self.info.get('w_bit', 4), q_config=q_config, init_only=True)

        if self.gpu_split != None:
            print('Loading big model with gpu_count', gpu_split)
            max_memory = {0:"18GiB", "cpu":"99GiB"} if gpu_split == "0,cpu" else { 0:"18GiB", 1:"22GiB" }
            device_map = infer_auto_device_map(model,
                                               no_split_module_classes=["DecoderLayer"],
                                               max_memory=max_memory)
            if device_map['lm_head'] == 'cpu': device_map['lm_head'] = 0
            print(device_map)
        else:
            device_map = 'balanced'
        
        self.model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map)

        print(f"Model loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")

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
        answer = self.tokenizer.decode(sample[0]).replace(prompt, '').replace('<|endoftext|>','').replace('</s>','').replace('<s>','')
        return answer, self.info

def interview_run(runtime, generate, interview, params_json, output_template, batch = False):
    if batch:
        print(f"Running batch of {len(interview)} prompts")
        prompts = [q['prompt'] for q in interview]
        answers, model_info = generate(prompts, params=params_json)
        print('Local model info:', model_info)
    else:
        answers = []
        model_info = None
        for idx, question in enumerate(interview):
            print(f"{idx+1}/{len(interview)} {question['name']} {question['language']}")

            # generate the answer
            result, info = generate(question['prompt'], params=params_json)

            # save for later
            if model_info is None:
                model_info = info
                print('Local model info:', model_info)

            # optional output template
            answer = output_template.render(**question, Answer=result) if output_template else result
            answers.append(answer)

            print()
            print(answer)
            print()

    results = []
    for idx, question in enumerate(interview):

        if batch:
            print()
            print(answers[idx])
            print()

        result = question.copy()
        result['answer'] = answers[idx]
        result['params'] = model_info['sampling_params']
        result['model'] = model_info['model_name']
        result['runtime'] = runtime
        results.append(result)

    return results, model_info

def download_safetensors(model_name):
    from huggingface_hub import snapshot_download, HfApi

    api = HfApi()
    files = api.list_files_info(model_name)
    
    search_list = ["safetensors"]
    found_safetensors = False
    for file_info in files:
        for needle in search_list:
            if file_info.rfilename.find(needle) != -1:
                found_safetensors = True
                break

    ignore_patterns = ["*.bin*"] if found_safetensors else []

    import os    
    if os.getenv('HF_HUB_ENABLE_HF_TRANSFER') != "1":
        print('WARING: You should set HF_HUB_ENABLE_HF_TRANSFER=1 and pip install hf-transfer for faster downloads')
    else:
        print('FAST downloading', model_name, 'found_safetensors=',found_safetensors)
    snapshot_download(model_name, ignore_patterns=ignore_patterns)

def main(input: str, params: str, model_name: str, runtime: str, info: str = "{}", iterations: int = 1, gpusplit: str = "", templateout: str = ""):
    from prepare import save_interview

    download_safetensors(model_name)

    gpu_split = gpusplit if gpusplit != '' else None
    model_info = json.loads(info)

    if runtime == 'transformers':
        model = InterviewTransformers(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'vllm':
        model = InterviewVLLM(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'autogptq':
        model = InterviewAutoGPTQ(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'exllama':
        model = InterviewExllama(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'awq':
        model = InterviewAWQ(model_name, model_info, gpu_split=gpu_split)
    else:
        raise Exception('Unknown runtime '+runtime)
    
    model.load()

    tasks = []
    for param_file in params.split(','):
        for input_file in input.split(','):
            tasks.append((param_file, input_file))

    output_template = Template(open(templateout).read()) if templateout else None

    for param_file, input_file in tasks:
      interview = [json.loads(line) for line in open(input_file)]
      params_json = json.load(open(param_file,'r'))

      for iter in range(iterations):
        results, remote_info = interview_run(runtime, model.generate, interview, params_json, output_template, batch=model.batch)
        save_interview(input_file, templateout if templateout else 'none', param_file, remote_info['model_name'], results)

if __name__ == "__main__":
    import fire
    fire.Fire(main)