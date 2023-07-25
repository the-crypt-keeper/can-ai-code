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
    def __init__(self, model_name, model_info = {}, quant = QUANT_FP16):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        print('Remote model', self.model_name, ' info', self.info)

        torch_dtype = torch.float32 if self.quant == QUANT_FP32 else torch.float16
        quantization_config = BitsAndBytesConfig(load_in_8bit = self.quant == QUANT_INT8,
                                                 load_in_4bit = self.quant == QUANT_FP4,
                                                 bnb_4bit_quant_type = "fp4")
        
        t0 = time.time()
        print('Loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, **self.info.get('tokenizer_args', {}))
        print('Loading model...')
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
        answer = self.tokenizer.decode(sample[0]).replace(prompt, '').replace('<|endoftext|>','').replace('</s>','')
        return answer, self.info

####################
##  vLLM Adapter  ##
####################

class InterviewVLLM:
    def __init__(self, model_name, model_info = {}, quant = QUANT_FP16):
        self.model_name = model_name
        self.info = model_info
        self.quant = quant

        self.batch = True

    def load(self):
        from vllm import LLM

        print('Remote model', self.model_name, ' info', self.info)

        t0 = time.time()
        print('Starting up...')
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
    
def interview_run(generate, interview, params_json, output_template, batch = False):
    if batch:
        print(f"Running batch of {len(interview)} prompts")
        prompts = [q['prompt'] for q in interview]
        answers, model_info = generate.call(prompts, params=params_json)
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
        result['params'] = info['sampling_params']
        result['model'] = info['model_name']
        result['runtime'] = 'transformers'
        results.append(result)

    return results, model_info

def main(input: str, params: str, model_name: str, iterations: int = 1, runtime: str = "transformers", templateout: str = ""):
    from prepare import save_interview

    if runtime == 'transformers':
        model = InterviewTransformers(model_name, {})
    elif runtime == 'vllm':
        model = InterviewVLLM(model_name, {})
    else:
        raise Exception('Unknown runtime '+runtime)
    
    model.load()

    interview = [json.loads(line) for line in open(input)]
    params_json = json.load(open(params,'r'))
    output_template = Template(open(templateout).read()) if templateout else None

    for iter in range(iterations):
        results, remote_info = interview_run(model.generate, interview, params_json, output_template, batch=model.batch)
        save_interview(input, templateout if templateout else 'none', params, remote_info['model_name'], results)

if __name__ == "__main__":
    import fire
    fire.Fire(main)