import json
import requests
import os
import time
from jinja2 import Template
import yaml
import argparse

# WARNING: This script is biased towards StarCoder interview challenges since it uses FIM prompting.
# tiny-interview.yml example at https://github.com/the-crypt-keeper/tiny_starcoder/blob/can-ai-code/tiny-interview.yml

parser = argparse.ArgumentParser(description='Interview executor for LangChain')
parser.add_argument('--tinyinterview', type=str, default='../tiny_starcoder/tiny-interview.yml', help='path to tiny-interview.yml from prepare stage')
parser.add_argument('--model', type=str, default='bigcode/starcoder', help='model to use')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
args = parser.parse_args()

headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
API_URL = "https://api-inference.huggingface.co/models/"+args.model
def query(payload):
    tries = 0
    while tries < 5:
        tries += 1
        response = requests.request("POST", API_URL, headers=headers, json=payload)
        res = {}
        try:
            res = json.loads(response.content.decode("utf-8"))
        except Exception as e:
            print('JSON decoder failed:', response.content.decode("utf-8"))
            time.sleep(1)
            continue
        if not isinstance(res, list):
            try:
                print('Generation error:', res['error'])
            except:
                print('Something weird went wrong', res)
            time.sleep(1)
            continue
    return res

input_template = Template("""<fim_prefix>def {{Signature}}:
    '''a function {{Input}} that returns {{Output}}'''
    <fim_suffix>

# another function
<fim_middle>""")

input_fact_template = Template("""<fim_prefix>def {{Signature}}:
    '''a function {{Input}} that returns {{Output}}, given {{Fact}}'''
    <fim_suffix>

# another function
<fim_middle>""")
                                   
output_template = Template("""def {{Signature}}:
    '''a function {{Input}} that computes {{Output}}'''
    {{Answer}}""")

interview = yaml.safe_load(open(args.tinyinterview))
for name, challenge in interview.items():
    
    challenge['name'] = name
    input = input_template.render(**challenge) if not challenge.get('Fact') else input_fact_template.render(**challenge)
    
    # for parameters, see https://huggingface.github.io/text-generation-inference/ GenerateParameters struct
    data = query(
        {
            "inputs": input,
            "parameters": {"temperature": 0.2, "top_k": 50, "top_p": 0.1, "max_new_tokens": 512, "repetition_penalty": 1.17},
        }
    )
    result = data[0]['generated_text']

    result = result.replace(input, '').replace('<|endoftext|>','')
    
    output = output_template.render(**challenge, Answer=result)

    print()
    print(output)
    print()

    with open(f"{args.outdir}/{name}.txt", "w") as f:
        f.write(output)
