#!/usr/bin/env python3
import time
import json
import os
from huggingface_hub import hf_hub_download, HfApi
from jinja2 import Template
from typing import List
from copy import copy
from transformers import AutoTokenizer

#########################################
##  Transformers/BitsAndBytes Adapter  ##
#########################################
   
QUANT_FP32 = 0
QUANT_FP16 = 1
QUANT_INT8 = 10
QUANT_FP4  = 20
QUANT_NF4  = 21

quant_suffix = {}
quant_suffix[QUANT_FP32] = 'fp32'
quant_suffix[QUANT_FP16] = 'fp16'
quant_suffix[QUANT_INT8] = 'int8'
quant_suffix[QUANT_FP4] = 'fp4'
quant_suffix[QUANT_NF4] = 'nf4'

hf_api = HfApi()
def hf_list_files(model, revision):
    return hf_api.list_repo_files(model, revision=revision)

class InterviewTransformers:
    def __init__(self, model_name, model_info = {}, quant = QUANT_FP16, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig, pipeline
        import torch

        # the gptq loader has a bug where it tries to re-download things if this is enabled
        import os
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

        # the gptq loader does not support accelerate, it uses optimum instead
        use_accelerate = self.info.get('accelerate', True)
        if 'gptq' in self.model_name.lower():
            use_accelerate = False
        # aqlm doesn't use accelerate either
        if 'aqlm' in self.model_name.lower():
            use_accelerate = False
        use_pipeline = self.info.get('pipeline', False)

        print('Remote model', self.model_name, ' info', self.info, 'use_accelerate', use_accelerate, 'use_pipeline', use_pipeline)

        t0 = time.time()
        tokenizer_model = self.info.get('tokenizer', self.model_name)
        print('Loading tokenizer',tokenizer_model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True, **self.info.get('tokenizer_args', {}))

        if self.info.get('dtype') == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif self.info.get('dtype') == 'float16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32 if self.quant == QUANT_FP32 else torch.bfloat16
            
        quantization_config = BitsAndBytesConfig(load_in_8bit = self.quant == QUANT_INT8,
                                                load_in_4bit = self.quant in [QUANT_FP4, QUANT_NF4],
                                                bnb_4bit_quant_type = "nf4" if self.quant == QUANT_NF4 else "fp4")
        if quantization_config.load_in_4bit: quantization_config.bnb_4bit_compute_dtype = torch.float16
        if self.quant == QUANT_FP16: quantization_config = None
               
        if use_pipeline:
            print('Loading model with pipeline...')
            self.pipeline = pipeline("text-generation", model=self.model_name, model_kwargs={"torch_dtype": torch.float16 })
            self.model = None
            mem_usage = self.pipeline.model.get_memory_footprint()
        elif use_accelerate:
            print('Loading model with accelerate...')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch_dtype, quantization_config=quantization_config, revision=self.info.get('revision',None), low_cpu_mem_usage=True, trust_remote_code=True)
            #self.model = self.model.to('cuda:0').eval()
            mem_usage = self.model.get_memory_footprint()
        else:
            print('Loading model ...')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, revision=self.info.get('revision',None), torch_dtype="auto")
            self.model.cuda()
            mem_usage = self.model.get_memory_footprint()

        # if passed a path, take the last dir name otherwise replace / with -
        if self.model_name[0] == '/':
            split_path = self.model_name.split('/')
            self.info['model_name'] = split_path[-1] if split_path[-1].strip() != '' else split_path[-2]
        else:
            self.info['model_name'] = self.model_name.replace('/','-')
        # add quant suffix
        if self.quant in quant_suffix:
            self.info['model_name'] += '-' + quant_suffix[self.quant]

        print(f"Model {self.info['model_name']} loaded in {time.time() - t0:.2f}s used {mem_usage/1024/1024:.2f}MB of memory")        

    def generate(self, prompt, params, gen_args = {}):
        t0 = time.time()
        
        from transformers import GenerationConfig

        generate_args = copy(self.info['generate_args']) if 'generate_args' in self.info else {}
        for k,v in gen_args.items():
            generate_args[k] = v

        if params.get('do_sample') is None: params['do_sample'] = True
        try:            
            generation_config, unused_kwargs = GenerationConfig.from_pretrained(
                self.model_name, **params, return_unused_kwargs=True
            )
        except Exception as e:
            print('WARNING: generate config could not be auto-loaded from model:', str(e))
            generation_config = GenerationConfig(**params)

        original_eos_token_id = generation_config.eos_token_id
            
        if self.info.get('eos_token_id') is not None:
            generation_config.eos_token_id = [self.info.get('eos_token_id')]
            if original_eos_token_id is not None: generation_config.eos_token_id += [original_eos_token_id]
        self.info['sampling_params'] = str(generation_config)
        # print('sampling_params', self.info['sampling_params'])
        
        eos_str_list = []

        if 'stop_seq' in generate_args:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_texts: List[str], *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.tokenizer = tokenizer
                    self.input_length = None
                    self.stop_texts = stop_texts

                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    decoded = self.tokenizer.decode(input_ids[0])

                    if self.input_length is None:
                        self.input_length = len(decoded)
                        return False

                    for stop_seq in self.stop_texts:
                        if stop_seq in decoded[self.input_length:]:
                            return True
                        
                    return False
            
            eos_str_list += generate_args['stop_seq']
            generate_args['stopping_criteria'] = StoppingCriteriaList([StopSequenceCriteria(self.tokenizer, generate_args['stop_seq'])])
            del generate_args['stop_seq']
            
        if self.model is None:           
            answer = self.pipeline(prompt, **params)
        else:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')
            input_len = inputs.size()[-1]
            
            sample = self.model.generate(inputs, generation_config=generation_config, **generate_args)
            answer = self.tokenizer.decode(sample[0][input_len:], clean_up_tokenization_spaces=False, skip_special_tokens=True)

        if generation_config.eos_token_id is not None:
            eos_token_ids = generation_config.eos_token_id
            if not isinstance(eos_token_ids, list): eos_token_ids = [eos_token_ids]                
            for tmp_eos_token in eos_token_ids:
                eos_str_list.append(self.tokenizer.decode([tmp_eos_token]))
                
        for tmp_eos_str in eos_str_list:
            answer = answer.replace(tmp_eos_str, '')

        t1 = time.time()
        output_len = len(sample[0])-input_len
        speed = output_len / (t1-t0)
        print(f"Generated {output_len} tokens in {t1-t0}s speed {speed:.2f} tok/sec")

        return answer, self.info

###########################
##  ctranslate2 Adapter  ##
###########################
class InterviewCtranslate2:
    def __init__(self, model_name, model_info = {}, quant = QUANT_FP16, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = True

    def load(self):
        from hf_hub_ctranslate2 import GeneratorCT2fromHfHub

        print('Remote model', self.model_name, ' info', self.info)

        t0 = time.time()
        print('Loading model...')
        self.model = GeneratorCT2fromHfHub(model_name_or_path=self.model_name, device="cuda", compute_type="int8_float16")

        # if passed a path, take the last dir name otherwise replace / with -
        if self.model_name[0] == '/':
            self.info['model_name'] = self.model_name.split('/')[-1]
        else:
            self.info['model_name'] = self.model_name.replace('/','-')
            
        print(f"Model {self.info['model_name']} loaded in {time.time() - t0:.2f}s")        

    def generate(self, prompts, params):
        model_params = {
            'max_length': params.get('max_new_tokens', 512),
            'sampling_temperature': params.get('temperature', 1.0),
            'sampling_topk': params.get('topk', 50),
            'sampling_topp': params.get('topp', 1.0),
            'repetition_penalty': params.get('repetition_penalty', 1.0),
            'num_hypotheses': params.get('num_beams', 1)
        }
        self.info['sampling_params'] = model_params

        if isinstance(prompts, list):
            text=['<s>'+x for x in prompts]
        else:
            text=['<s>'+prompts]

        token_streams = [[] for x in text]
        stop_seqs = params.get('stop_seqs', [])
        def callback(x):
            stream = token_streams[x.batch_id]
            stream += [x.token_id]

            for stop_seq in stop_seqs:
                if len(stream) < len(stop_seq):
                    continue
                if stream[len(stream)-len(stop_seq):] == stop_seq:
                    print(f"Batch {x.batch_id} stop_seq terminated at step {x.step}")
                    return True
                
            if x.is_last:
                print(f"Batch {x.batch_id} completed at step {x.step}")
            return False
        
        answers = self.model.generate(
            text=text,
            include_prompt_in_result=False,
            callback=callback,
            **model_params
        )

        return answers if len(answers)>1 else answers[0], self.info
    
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
        self.model = AutoGPTQForCausalLM.from_quantized(self.model_name, device="cuda:0", use_triton=False, revision=self.info.get('revision',None), use_safetensors=self.info.get('model_safetensors', True), trust_remote_code=True)
    
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
            "top_k": params.get('top_k', -1),
            "top_p": params.get('top_p', 1.0),
            "max_new_tokens": params.get('max_new_tokens', 512)
        }
        self.info['sampling_params'] = sampling_params

        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        sample = self.model.generate(input_ids=tokens, do_sample=True, **sampling_params)
        answer = self.tokenizer.decode(sample[0]).replace(prompt, '').replace('<|endoftext|>','').replace('</s>','').replace('<s>','')
        return answer, self.info

########################
##  exllama2 Adapter  ##
########################
class InterviewExllama2:
    def __init__(self, model_name, model_info = {}, gpu_split=None, token_healing = False, cache_4bit = False):
        self.model_name = model_name
        self.info = model_info
        
        self.token_healing = token_healing
        self.cache_4bit = cache_4bit
        
        self.batch_size = self.info.get('batch_size', 1)

        self.tokenizer = None
        self.model = None
        self.cache = None

        self.info['model_name'] = self.model_name + '-' + self.info.get('revision','main')
        #if '70B' in model_name: self.info['low_mem'] = True

    def load(self):
        print("Starting load..")
        # import sys
        # sys.path += ["/repositories/exllamav2","../exllamav2"]
        import os
        
        # monkey-patch a fix for https://github.com/the-crypt-keeper/can-ai-code/issues/114
        from exllamav2 import compat
        original_test_gpu_peer_copy = compat.test_gpu_peer_copy
        def safe_test_gpu_peer_copy(x,y):
            try:
                return original_test_gpu_peer_copy(x,y)
            except Exception as e:
                print('test_gpu_peer_copy() failed: ', str(e))
                return False
        compat.test_gpu_peer_copy = safe_test_gpu_peer_copy        

        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Cache,
            ExLlamaV2Cache_Q4,
            ExLlamaV2Tokenizer,
        )

        config_path = hf_hub_download(repo_id=self.model_name, revision=self.info.get('revision',None), filename="config.json")

        config = ExLlamaV2Config()
        config.model_dir = os.path.dirname(config_path)

        print('Starting up...')
        if self.info.get('low_mem', False): config.set_low_mem()
        config.prepare()

        print("Loading tokenizer...")
        self.tokenizer = ExLlamaV2Tokenizer(config)

        print("Loading model...")
        self.model = ExLlamaV2(config)
        if self.cache_4bit:
            print("Using 4-bit KV cache...")
            self.cache = ExLlamaV2Cache_Q4(self.model, max_seq_len=2048, lazy=True, batch_size = self.batch_size)
        else:
            self.cache = ExLlamaV2Cache(self.model, max_seq_len=2048, lazy=True, batch_size = self.batch_size)
        self.model.load_autosplit(self.cache, progress = True)

        if self.info.get('eos_token_id'):
            self.tokenizer.eos_token_id = self.info.get('eos_token_id')
            print("overide stop_token:", self.tokenizer.eos_token_id)
            
    def generate(self, prompt, params):
        from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
        
        generator = ExLlamaV2DynamicGenerator(self.model, self.cache, self.tokenizer)        
        
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = params.get('temperature', 1.0)
        settings.top_k = params.get('top_k', -1)
        settings.top_p = params.get('top_p', 0.0)
        settings.token_repetition_penalty = params.get('repetition_penalty', 1.0)

        self.info['sampling_params'] = str(settings.__dict__)

        max_new_tokens = params.get('max_new_tokens', 512)
        stop_text = self.info.get('generate_args',{}).get('stop_seq', [])
        stop_condition_list = self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]
        stop_condition_list += stop_text
        
        prompts = prompt if isinstance(prompt, list) else [prompt]
        
        print('Starting batch generation, please wait..')
        answers = generator.generate(
            prompt = prompts,
            max_new_tokens = max_new_tokens,
            stop_conditions = stop_condition_list,
            gen_settings = settings,            
            token_healing = self.token_healing,
            encode_special_tokens = True,
            completion_only = True
        )
        
        if not isinstance(prompt, list): answers = answers[0]

        return answers, self.info

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
        from transformers import GenerationConfig
        import torch

        print('Remote model', self.model_name, ' info', self.info)

        t0 = time.time()
        quantization = None
        dtype = 'bfloat16'
        if 'awq' in self.model_name.lower(): 
            quantization = 'awq'
            dtype = 'float16'
        if 'gptq' in self.model_name.lower() or 'int4' in self.model_name.lower():
            quantization = 'gptq'
            dtype = 'float16'
        if 'sq-' in self.model_name.lower():
            quantization = 'squeezellm'
        if 'aqlm' in self.model_name.lower():
            quantization = 'aqlm'
            dtype = 'float16'
        
        tokenizer_mode = self.info.get('tokenizer_mode', 'auto')
        max_model_len = self.info.get('max_model_len', 2048)
        enforce_eager = self.info.get('enforce_eager', True)
        
        import os
        os.environ['VLLM_ATTENTION_BACKEND'] = self.info.get('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')

        gpu_memory_utilization = 0.95
        if self.gpu_split > 1:
            print('Starting in multi-gpu mode...')
            if quantization is not None and 'awq' in quantization: 
                gpu_memory_utilization = 0.9
            self.llm = LLM(model=self.model_name, revision=self.info.get('revision',None), quantization=quantization, tokenizer_mode=tokenizer_mode, dtype=dtype, max_model_len=max_model_len, tensor_parallel_size=self.gpu_split, trust_remote_code=True, enforce_eager=enforce_eager, gpu_memory_utilization=gpu_memory_utilization)
        else:
            print('Starting in single GPU mode..')
            self.llm = LLM(model=self.model_name, revision=self.info.get('revision',None), quantization=quantization, tokenizer_mode=tokenizer_mode, dtype=dtype, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=enforce_eager, gpu_memory_utilization=gpu_memory_utilization)
            
        eos_token_id = self.info.get('eos_token_id', None)
        if eos_token_id is not None:
            print('Override generate_args.eos_token_id = ', eos_token_id)
        else:
            generation_config = None
            try:
                generation_config, unused_kwargs = GenerationConfig.from_pretrained(self.model_name, return_unused_kwargs=True)
                if generation_config.eos_token_id is not None:
                    eos_token_id = generation_config.eos_token_id
                    print('Loaded eos_token_id from generation_config:', eos_token_id)
            except Exception as e:
                print('WARNING: generate config could not be auto-loaded from model:', str(e))

        if eos_token_id is None:
            self.stop_token_ids = []
        elif isinstance(eos_token_id,list):
            self.stop_token_ids = [int(x) for x in eos_token_id]
        else:
            self.stop_token_ids = [int(eos_token_id)]

        if self.model_name[0] == '/':
            split_path = self.model_name.split('/')
            self.info['model_name'] = split_path[-1] if split_path[-1].strip() != '' else split_path[-2]
        else:        
            self.info['model_name'] = self.model_name

        print(f"Model loaded in {time.time() - t0:.2f}s")   

    def generate(self, prompt, params):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            stop=self.info.get('generate_args', {}).get('stop_seq', []),
            stop_token_ids=self.stop_token_ids,
            temperature=params.get('temperature', 1.0),
            top_k=params.get('top_k', -1),
            top_p=params.get('top_p', 1.0),
            max_tokens=params.get('max_new_tokens', 512),
            repetition_penalty=params.get('repetition_penalty', 1.0)
        )
        result = self.llm.generate(prompt, sampling_params)
        self.info['sampling_params'] = str(sampling_params)

        answers = []
        if isinstance(prompt, list):            
            for i in range(len(prompt)):
                for r in result:
                    if r.prompt == prompt[i]:
                        answers.append(r.outputs[0].text.replace('</s>','').replace('<|endoftext|>',''))
                        break
        else:
            answers = result[0].outputs[0].text.replace('</s>','').replace('<|endoftext|>','')

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

        self.batch = False

    def load(self):
        import torch
        from awq.quantize.quantizer import real_quantize_model_weight
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

        # Config
        print('Starting up...')
        base_model = self.info.get('base_model', self.model_name)
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Model
        t0 = time.time()

        files = hf_list_files(self.model_name, self.info.get('revision',None))
        model_path = None
        search_list = [".index.json", ".pt", ".bin"]
        for file_info in files:
            for needle in search_list:
                if file_info.find(needle) != -1:
                    model_path = hf_hub_download(repo_id=self.model_name, filename=file_info.rfilename)
                    break

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)

        q_config = { "zero_point": True, "q_group_size": self.info.get('q_group_size', 128) }
        real_quantize_model_weight(model, w_bit=self.info.get('w_bit', 4), q_config=q_config, init_only=True)

        if self.gpu_split > 1 or self.info.get('big_model'):
            print('Loading big model with gpu_count', self.gpu_split)
            if self.info.get('big_model'):
                max_memory = {0:"18GiB", "cpu":"99GiB"}
            else:
                # TODO: this assumes 24GB GPUs
                max_memory = { device_id:"18GiB" for device_id in range(0, self.gpu_split) }
                
            device_map = infer_auto_device_map(model,
                                               no_split_module_classes=["DecoderLayer"],
                                               max_memory=max_memory)
            
            # TODO: this hack forces lm_head to the GPU
            if device_map.get('lm_head') == 'cpu': device_map['lm_head'] = 0
        else:
            device_map = 'balanced'
        
        self.model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map)

        self.info['model_name'] = self.model_name

        print(f"Model loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")

    def generate(self, prompt, params):
        input = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input.input_ids.to('cuda')
        attention_mask = input.attention_mask.to('cuda')
        sampling_params = {
            'do_sample': True,
            'temperature': params.get('temperature', 1.0),
            'max_length': params.get('max_new_tokens', 512),
            'top_k': params.get('top_k', -1),
            'top_p': params.get('top_p', 1.0),
            'repetition_penalty': params.get('repetition_penalty', 1.0)
        }
        sample = self.model.generate(input_ids, attention_mask=attention_mask, use_cache=True, eos_token_id=self.tokenizer.eos_token_id, **sampling_params)
        self.info['sampling_params'] = sampling_params
        answer = self.tokenizer.decode(sample[0]).replace(prompt, '').replace('<|endoftext|>','').replace('</s>','').replace('<s>','')
        return answer, self.info

###################
##  HQQ Adapter  ##
###################
class InterviewHQQ:
    def __init__(self, model_name, model_info = {}, quant = None, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant
        self.batch = False

    def load(self):
        import torch
        from transformers import AutoTokenizer
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.utils.patching import patch_linearlayers, patch_add_quant_config, prepare_for_inference
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
        from hqq.utils.generation_hf import HFGenerator

        # Config
        print('Starting up...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print('Loading model...')
        t0 = time.time()
        compute_dtype = torch.float16 #bfloat16 for torchao, float16 for bitblas
        cache_dir = '.'
        self.model     = AutoHQQHFModel.from_quantized(self.model_name, cache_dir=cache_dir, compute_dtype=compute_dtype)
        self.info['model_name'] = self.model_name
        print(f"Model loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")
        
        print('Preparing backend..')
        quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1)
        patch_linearlayers(self.model, patch_add_quant_config, quant_config)

        HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)
        prepare_for_inference(self.model) #default backend
        #prepare_for_inference(self.model, backend="torchao_int4") #need bfloat16
        #prepare_for_inference(self.model, backend="bitblas")
        
        print('Compiling generator...')
        self.gen = HFGenerator(self.model, self.tokenizer, max_new_tokens=1024, do_sample=False, compile="partial")
        
        print('Loading complete.')

    def generate(self, prompt, params):
        print('Generating, please wait...')
        self.info['sampling_params'] = {'do_sample': False }
        result = self.gen.generate(prompt, use_chat_template = False, print_tokens = True)
        answer = result['output_text']
        return answer, self.info

#########################
##  QuipSharp Adapter  ##
#########################
class InterviewQuipSharp:
    def __init__(self, model_name, model_info = {}, quant = None, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant
        self.batch = False

    def load(self):
        import sys
        sys.path += ["/repositories/quip-sharp","../quip-sharp"]
        
        import transformer_engine
        import transformer_engine_extensions
        
        import torch
        from transformers import AutoTokenizer
        from lib.utils.unsafe_import import model_from_hf_path

        print('Starting up...')
        
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        
        model_path = os.path.dirname(hf_hub_download(repo_id=self.model_name, filename='model.safetensors'))

        print('Loading model...')
        t0 = time.time()        
        self.model, model_str = model_from_hf_path(model_path, use_cuda_graph=self.info.get('use_cuda_graph', False), use_flash_attn=self.info.get('use_flash_attn', False))
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")

        self.info['model_name'] = self.model_name       

    def generate(self, prompt, params):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        sample = self.model.generate(input_ids=inputs['input_ids'].cuda(),
                                 attention_mask=inputs['attention_mask'].cuda(),
                                 max_length=2048,
                                 penalty_alpha=0.6,
                                 top_k=-1,
                                 use_cache=True,
                                 return_dict_in_generate=True).sequences[0]
        
        self.info['sampling_params'] = {}       
        answer = self.tokenizer.decode(sample[0][len(inputs[0]):], skip_special_tokens=True)
        return answer, self.info
    
def interview_run(runtime, generate, interview, params_json, output_template, batch = False, quiet = False):
    if batch:
        if not quiet: print(f"Running batch of {len(interview)} prompts")
        prompts = [q['prompt'] for q in interview]
        answers, model_info = generate(prompts, params=params_json)
        if not quiet: print('Local model info:', model_info)
    else:
        answers = []
        model_info = None
        for idx, question in enumerate(interview):
            if not quiet: print(f"{idx+1}/{len(interview)} {question['name']} {question['language']}")

            # generate the answer
            result, info = generate(question['prompt'], params=params_json)

            # save for later
            if model_info is None:
                model_info = info
                if not quiet: print('Local model info:', model_info)

            # optional output template
            answer = output_template.render(**question, Answer=result) if output_template else result
            answers.append(answer)

            if not quiet: 
                print()
                print(answer)
                print()

    results = []
    for idx, question in enumerate(interview):

        answer = answers[idx]
        if batch:
            answer = output_template.render(**question, Answer=answer) if output_template else answer
            print()
            print(answer)
            print()        

        result = question.copy()
        result['answer'] = answer
        result['params'] = model_info['sampling_params']
        result['model'] = model_info['model_name']
        result['runtime'] = runtime
        results.append(result)

    return results, model_info

def download_safetensors(model_name, revision=None):
    from huggingface_hub import snapshot_download, HfApi

    files = hf_list_files(model_name, revision)
    
    search_list = ["safetensors"]
    found_safetensors = False
    for file_info in files:
        for needle in search_list:
            if file_info.find(needle) != -1:
                found_safetensors = True
                break

    ignore_patterns = ["*.bin*"] if found_safetensors else []

    import os    
    if os.getenv('HF_HUB_ENABLE_HF_TRANSFER') != "1":
        print('WARING: You should set HF_HUB_ENABLE_HF_TRANSFER=1 and pip install hf-transfer for faster downloads')
    else:
        print('FAST downloading', model_name, 'revision=',revision, 'found_safetensors=',found_safetensors)

    while True:
        try:
            snapshot_download(model_name, ignore_patterns=ignore_patterns, resume_download=True, revision=revision)
        except KeyboardInterrupt:
            print('Download aborted')
            exit(1)
        except Exception as e:
            print('Download problem: ', e)
            continue
        break
    
#################################
##  Mistral-Inference Adapter  ##
#################################
def is_torchrun() -> bool:
    required_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    return all(var in os.environ for var in required_vars)

class InterviewMistral:
    def __init__(self, model_name, model_info = {}, gpu_split=None, token_healing = False, cache_8bit = False):
        self.model_name = model_name
        self.info = model_info
        
        self.tokenizer = None
        self.model = None
        self.batch = False

        self.info['model_name'] = self.model_name.split('/')[-1]

    def load(self):
        print("Starting load..")
        config_path = self.model_name # hf_hub_download(repo_id=self.model_name, revision=self.info.get('revision',None), filename="params.json")
        
        from mistral_inference.model import Transformer
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        import torch
        
        if is_torchrun():
            torch.distributed.init_process_group()
            torch.cuda.set_device(torch.distributed.get_rank())
            num_pipeline_ranks = torch.distributed.get_world_size()
        else:
            num_pipeline_ranks = 1
                
        print("Loading model...")
        dtype = torch.bfloat16 if self.info.get('bf16', False) else torch.float16
        self.tokenizer = MistralTokenizer.from_file(f"{config_path}/tokenizer.model.v3")
        self.model = Transformer.from_folder(config_path, num_pipeline_ranks=num_pipeline_ranks, dtype=dtype)
            
    def generate(self, prompt, params):
        from mistral_inference.generate import generate
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest        
        
        eos_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        if self.info.get('eos_token_id'):
            eos_id = self.info.get('eos_token_id')
            # print("overide stop_token:", eos_id)
            
        self.info['sampling_params'] = {
            'max_tokens': params.get('max_new_tokens'),
            'temperature': params.get('temperature', 0.0)
        }
                
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], self.model, eos_id=eos_id, **self.info['sampling_params'])
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return result, self.info

def load_runtime(model_name, model_info, runtime, quant, num_gpus):
    if quant:
        quant_id = None
        for k,v in quant_suffix.items():
            if v == quant:
                quant_id = k
        if quant_id is None:
            raise Exception("quant "+quant+" not found")
        else:
            print("quant:", quant_id)
    else:
        quant_id = QUANT_FP16
            
    if runtime == 'transformers':
        model = InterviewTransformers(model_name, model_info, quant=quant_id)
    elif runtime == 'vllm':
        model = InterviewVLLM(model_name, model_info, gpu_split=num_gpus)
    elif runtime == 'autogptq':
        model = InterviewAutoGPTQ(model_name, model_info)
    elif runtime[0:8] == 'exllama2':
        model = InterviewExllama2(model_name, model_info, token_healing=True)
    elif runtime == 'awq':
        model = InterviewAWQ(model_name, model_info, gpu_split=num_gpus)
    elif runtime == 'hqq':
        model = InterviewHQQ(model_name, model_info)
    elif runtime == 'ctranslate2':
        model = InterviewCtranslate2(model_name, model_info)
    elif runtime == 'mistral':
        model = InterviewMistral(model_name, model_info)
    else:
        raise Exception('Unknown runtime '+runtime)
    
    return model


def main(model: str, runtime: str, input: str = "", interview: str = "senior", params: str = "", templateout: str = "", revision: str = "", info: str = "{}"):
    from prepare import save_interview, prepare_interview
    import torch
    
    if params == "": params = "params/greedy-hf.json" if runtime == "transformers" else "params/greedy-openai.json"
    quant = 'fp16'
        
    if not os.path.exists(model):
        download_safetensors(model, revision if revision else None)
        
    model_info = json.loads(info) if isinstance(info, str) else info
    if revision: model_info['revision'] = revision

    # if completion or stop != '':
    #     ga = model_info.get('generate_args', {})
    #     ga['stop_seq'] = ga.get('stop_seq', [])
    #     if completion:
    #         ga['stop_seq'] += ["\n#","\n//"]
    #     if stop != '':
    #         ga['stop_seq'] += stop
    #     model_info['generate_args'] = ga    

    num_gpus = torch.cuda.device_count()
    wrapper = load_runtime(model, model_info, runtime, quant, num_gpus)
    wrapper.load()

    interviews = []
    if input != "":
        for input_file in input.split(','):
            interview = [json.loads(line) for line in open(input_file)]
            interviews.append( (input_file, interview) )
            print(f"Loaded {len(interview)} questions from {input_file}.")
    elif interview != "":
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, revision=revision if revision else None)
        for interview_name in interview.split(','):
            language = "python,javascript"
            template_name = "chat-simple"
            message_template = [{'role': 'user', 'content': Template("Write a {language} function {Signature} {Input} that returns {Output}".replace('{','{'+'{').replace('}','}'+'}'))}]
            output_filename, interview = prepare_interview(interview_name, language, message_template, template_name, tokenizer)
            interviews.append( (output_filename, interview) )
            print(f"Expanded {len(interview)} questions from {interview_name}.")
    else:
        raise Exception("Please provide either --input or --interview")

    main_process = True
    if is_torchrun():        
        main_process = torch.distributed.get_rank() == 0

    output_template = Template(open(templateout).read()) if templateout else None
    params_json = json.load(open(params,'r'))

    for input_file, interview in interviews:
        if main_process:
            print("Starting", model, "param_file=", params, "input_file=", input_file, "templateout=", templateout)

        results, remote_info = interview_run(runtime, wrapper.generate, interview, params_json, output_template, batch=wrapper.batch, quiet=not main_process)

        if main_process:
            save_interview(input_file, templateout if templateout else 'none', params, remote_info['model_name'], results)
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)
