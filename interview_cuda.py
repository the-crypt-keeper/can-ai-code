#!/usr/bin/env python3
import time
import json
from jinja2 import Template
from typing import List
from copy import copy

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

class InterviewTransformers:
    def __init__(self, model_name, model_info = {}, quant = QUANT_FP16, gpu_split = None):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig
        import torch

        # the gptq loader has a bug where it tries to re-download things if this is enabled
        import os
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

        # the gptq loader does not support accelerate, it uses optimum instead
        use_accelerate = self.info.get('accelerate', True)
        if 'gptq' in self.model_name.lower():
            use_accelerate = False

        print('Remote model', self.model_name, ' info', self.info, 'use_accelerate', use_accelerate)

        t0 = time.time()
        tokenizer_model = self.info.get('tokenizer', self.model_name)
        print('Loading tokenizer',tokenizer_model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True, **self.info.get('tokenizer_args', {}))

        torch_dtype = torch.float32 if self.quant == QUANT_FP32 else torch.float16
        quantization_config = BitsAndBytesConfig(load_in_8bit = self.quant == QUANT_INT8,
                                                load_in_4bit = self.quant in [QUANT_FP4, QUANT_NF4],
                                                bnb_4bit_quant_type = "nf4" if self.quant == QUANT_NF4 else "fp4")
        if use_accelerate:
            print('Loading model with accelerate...')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch_dtype, quantization_config=quantization_config, revision=self.info.get('revision',None), trust_remote_code=True)
        else:
            print('Loading model ...')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype="auto")
            self.model.cuda()

        # if passed a path, take the last dir name otherwise replace / with -
        if self.model_name[0] == '/':
            self.info['model_name'] = self.model_name.split('/')[-1]
        else:
            self.info['model_name'] = self.model_name.replace('/','-')
        # add quant suffix
        if self.quant in quant_suffix:
            self.info['model_name'] += '-' + quant_suffix[self.quant]

        print(f"Model {self.info['model_name']} loaded in {time.time() - t0:.2f}s used {self.model.get_memory_footprint()/1024/1024:.2f}MB of memory")        

    def generate(self, prompt, params, gen_args = {}):
        from transformers import GenerationConfig

        generate_args = copy(self.info['generate_args']) if 'generate_args' in self.info else {}
        for k,v in gen_args.items():
            generate_args[k] = v

        try:
            generation_config, unused_kwargs = GenerationConfig.from_pretrained(
                self.model_name, do_sample = True, **params, return_unused_kwargs=True
            )
        except Exception as e:
            print('WARNING: generate config could not be auto-loaded from model:', str(e))
            generation_config = GenerationConfig(do_sample = True, **params)

        if not generation_config.eos_token_id:
            generation_config.eos_token_id = self.info.get('eos_token_id', self.tokenizer.eos_token_id)
        self.info['sampling_params'] = str(generation_config)

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
                
            generate_args['stopping_criteria'] = StoppingCriteriaList([StopSequenceCriteria(self.tokenizer, generate_args['stop_seq'])])
            del generate_args['stop_seq']
            
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        input_len = inputs.size()[-1]
        sample = self.model.generate(inputs, generation_config=generation_config, **generate_args)
        answer = self.tokenizer.decode(sample[0][input_len:], clean_up_tokenization_spaces=False)
       
        eos_list = [ '<|end|>', '<|endoftext|>', '<|endofmask|>', '</s>', '<s>', '<EOT>', '<empty_output>', '<|im_end|>', '[EOS]' ]
        eos_token = self.tokenizer.decode([generation_config.eos_token_id])
        if not eos_token in eos_list: eos_list += [eos_token]
        if 'stopping_criteria' in generate_args: eos_list += generate_args['stopping_criteria'][0].stop_texts
        
        for eos in eos_list:
            answer = answer.replace(eos, '')

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
        sys.path += ["/repositories/exllama","../exllama"]
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
        files = api.list_files_info(self.model_name, revision=self.info.get('revision',None))
        model_path = None
        for file_info in files:
            if (file_info.rfilename.find(".safetensors") != -1):
                model_path = hf_hub_download(repo_id=self.model_name, revision=self.info.get('revision',None), filename=file_info.rfilename)
                break
        
        if model_path is None:
            raise Exception("Could not find safetensors.")
        else:
            print("Loading from", model_path)
        
        self.config = ExLlamaConfig(hf_hub_download(repo_id=self.model_name, filename="config.json"))
        self.config.model_path = model_path
        self.config.max_seq_len = self.info.get('max_seq_len', 2048)
        self.config.compress_pos_emb = self.info.get('compress_pos_emb', 1.0)

        if self.gpu_split is not None:
            self.config.set_auto_map(self.gpu_split)

        print('Loading model...')
        t0 = time.time()
        self.model = ExLlama(self.config)
        self.cache = ExLlamaCache(self.model)
        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.info['model_name'] = self.model_name

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

########################
##  exllama2 Adapter  ##
########################
class InterviewExllama2:
    def __init__(self, model_name, model_info = {}, gpu_split=None):
        self.model_name = model_name
        self.gpu_split = gpu_split
        self.info = model_info

        self.tokenizer = None
        self.model = None
        self.cache = None

        self.info['model_name'] = self.model_name + '-' + self.info.get('revision','main')
        #if '70B' in model_name: self.info['low_mem'] = True

    def load(self):
        import sys
        sys.path += ["/repositories/exllamav2","../exllamav2"]
        from huggingface_hub import hf_hub_download
        import os

        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Cache,
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
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=2048, lazy=True)
        if self.info.get('low_mem', False): 
            self.model.load()
        else:
            self.model.load_autosplit(self.cache)

    def generate(self, prompt, params):
        from exllamav2.generator import (
            ExLlamaV2BaseGenerator,
            ExLlamaV2Sampler,
        )
        import torch
        import random

        class ExLlamaV2CustomGenerator(ExLlamaV2BaseGenerator):
                def generate_simple(self, prompt: str or list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed = None,
                        token_healing = False,
                        encode_special_tokens = False,
                        decode_special_tokens = False,
                        loras = None,
                        stop_token = -1):

                    if stop_token == -1: stop_token = self.tokenizer.eos_token_id
                    #if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]
                    #if seed is not None: random.seed(seed)

                    # Tokenize input and produce padding mask if needed

                    # Auto-detect BOS from prompt
                    add_bos = False
                    if isinstance(prompt, str):
                        batch_size = 1
                        if prompt[0:3] == '<s>': 
                            add_bos = True
                            prompt = prompt[3:]
                    else:
                        batch_size = len(prompt)                        
                        if prompt[0][0:3] == '<s>': 
                            add_bos = True
                            prompt = [x[3:] for x in prompt]

                    ids = self.tokenizer.encode(prompt, encode_special_tokens = encode_special_tokens, add_bos = add_bos)
                    self.prompt_ids = ids.clone()

                    overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
                    if overflow > 0: ids = ids[:, overflow:]

                    mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

                    # Prepare for healing

                    unhealed_token = None
                    if ids.shape[-1] < 2: token_healing = False
                    if token_healing:
                        unhealed_token = ids[:, -1:]
                        ids = ids[:, :-1]

                    # Process prompt and begin gen

                    self._gen_begin_base(ids, mask, loras)

                    # Begin filters

                    id_to_piece = self.tokenizer.get_id_to_piece_list()
                    if unhealed_token is not None:
                        unhealed_token_list = unhealed_token.flatten().tolist()
                        heal = [id_to_piece[x] for x in unhealed_token_list]
                    else:
                        heal = None
                    gen_settings.begin_filters(heal)

                    # Generate tokens

                    batch_eos = [False] * batch_size

                    for i in range(num_tokens):

                        logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, input_mask = mask, loras = loras).float().cpu()
                        token, _, _ = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random(), self.tokenizer, prefix_token = unhealed_token)

                        eos = False
                        if stop_token is not None:
                            for b in range(batch_size):
                                if token[b, 0].item() == stop_token:
                                    batch_eos[b] = True
                                    if all(batch_eos): eos = True
                                if batch_eos[b]:
                                    token[b, 0] = self.tokenizer.pad_token_id

                        self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
                        gen_settings.feed_filters(token)

                        unhealed_token = None
                        if eos: break

                    # Decode
                    text = []
                    for i in range(batch_size):
                        prompt_len = self.prompt_ids[i,:].shape[-1]
                        text.append( self.tokenizer.decode(self.sequence_ids[i,prompt_len:], decode_special_tokens = decode_special_tokens) )

                    if isinstance(prompt, str): return text[0]
                    return text


        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = params.get('temperature', 1.0)
        settings.top_k = params.get('top_k', 1000)
        settings.top_p = params.get('top_p', 1.0)
        settings.token_repetition_penalty = params.get('repetition_penalty', 1.0)
        #settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        self.info['sampling_params'] = str(settings.__dict__)

        max_new_tokens = params.get('max_new_tokens', 512)

        generator = ExLlamaV2CustomGenerator(self.model, self.cache, self.tokenizer)
        generator.warmup()

        time_begin = time.time()

        output = generator.generate_simple(prompt, settings, max_new_tokens, seed = self.info.get('seed', 0), encode_special_tokens = False, decode_special_tokens = False)

        time_end = time.time()
        time_total = time_end - time_begin
        
        #print(f"Response generated in {time_total:.2f} seconds")

        return output, self.info

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
        quantization = 'awq' if 'awq' in self.model_name.lower() else None
        dtype = 'float16' if quantization == 'awq' else 'bfloat16'
        if self.gpu_split is not None:
            print('Starting in multi-gpu mode...')
            self.llm = LLM(model=self.model_name, quantization=quantization, dtype=dtype, max_model_len=4096, tensor_parallel_size=self.gpu_split, trust_remote_code=True)
        else:
            print('Starting in single GPU mode..')
            self.llm = LLM(model=self.model_name, quantization=quantization, dtype=dtype, max_model_len=4096, trust_remote_code=True)

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

        if eos_token_id is not None:
            self.llm.llm_engine.tokenizer.eos_token_id = int(eos_token_id)
        
        self.info['model_name'] = self.model_name

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

        self.batch = False

    def load(self):
        import torch
        from awq.quantize.quantizer import real_quantize_model_weight
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
        from huggingface_hub import hf_hub_download, HfApi

        # Config
        print('Starting up...')
        base_model = self.info.get('base_model', self.model_name)
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
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
            print('Loading big model with gpu_count', self.gpu_split)
            max_memory = {0:"18GiB", "cpu":"99GiB"} if self.gpu_split == "0,cpu" else { 0:"18GiB", 1:"22GiB" }
            device_map = infer_auto_device_map(model,
                                               no_split_module_classes=["DecoderLayer"],
                                               max_memory=max_memory)
            if device_map.get('lm_head') == 'cpu': device_map['lm_head'] = 0
            device_map = 'balanced'
            print(device_map)
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

    api = HfApi()
    files = api.list_files_info(model_name, revision=revision)
    
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

def main(input: str, params: str, model_name: str, runtime: str, info: str = "{}", iterations: int = 1, quant: str = "", gpusplit: str = "", templateout: str = "", revision: str = ""):
    from prepare import save_interview

    download_safetensors(model_name, revision if revision else None)

    gpu_split = gpusplit if gpusplit != '' else None
    model_info = json.loads(info) if isinstance(info, str) else info
    if revision: model_info['revision'] = revision

    if runtime == 'transformers':
        if quant:
            quant_id = None
            for k,v in quant_suffix.items():
                if v == quant:
                    quant_id = k
            if not quant_id:
                raise Exception("quant "+quant+" not found")
        else:
            quant_id = QUANT_FP16
        model = InterviewTransformers(model_name, model_info, gpu_split=gpu_split, quant=quant_id)
    elif runtime == 'vllm':
        model = InterviewVLLM(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'autogptq':
        model = InterviewAutoGPTQ(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'exllama':
        model = InterviewExllama(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'exllama2':
        model = InterviewExllama2(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'awq':
        model = InterviewAWQ(model_name, model_info, gpu_split=gpu_split)
    elif runtime == 'ctranslate2':
        model = InterviewCtranslate2(model_name, model_info, gpu_split=gpu_split)
    else:
        raise Exception('Unknown runtime '+runtime)
    
    model.load()

    tasks = []
    for param_file in params.split(','):
        for input_file in input.split(','):
            tasks.append((param_file, input_file))

    for param_file, input_pairs in tasks:
      insplit = input_pairs.split(':')
      input_file = insplit[0]
      templateout_file = insplit[1] if len(insplit)>1 else templateout

      interview = [json.loads(line) for line in open(input_file)]
      output_template = Template(open(templateout_file).read()) if templateout_file else None
      params_json = json.load(open(param_file,'r'))

      for iter in range(iterations):
        print("Starting", model_name, "iter=", iter, "param_file=", param_file, "input_file=", input_file, "templateout_file=", templateout_file)
        results, remote_info = interview_run(runtime, model.generate, interview, params_json, output_template, batch=model.batch)
        save_interview(input_file, templateout_file if templateout_file else 'none', param_file, remote_info['model_name'], results)

if __name__ == "__main__":
    import fire
    fire.Fire(main)