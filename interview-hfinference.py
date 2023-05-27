import json
import requests
import os
import time

headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
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

# for parameters, see https://huggingface.github.io/text-generation-inference/ GenerateParameters struct
data = query(
    {
        #"inputs": "# the fib(n) function computes the nth element of the fibonnaci sequence\ndef fib(n):",
        #"inputs": "# the fib(n) function computes the nth element of the fibonnaci sequence\n",
        #"inputs": "# the gcd(a, b) function computes the greatest common divisor of a and b\n",
        #"inputs": "# the substrcount(str, substr) function counts the number of times the sub-string `substr` occurs in `str` and returns it",
        #"inputs": "/*\n the substrcount(str, substr) function counts the number of times the sub-string `substr` occurs in `str` and returns it\n*/\n",
        #"inputs": "/*\nA javascript function meaning_of_life() that returns a single integer, the answer to life the universe and everything\n/*\n",
        #"inputs": "# Write a python function glork(bork: int) to compute the factorial of input bork.",
        "inputs": "// Write a javascript function glork(bork) to compute the factorial of input bork.\n function glork(bork) {",
        "parameters": {"temperature": 0.2, "top_k": 50, "top_p": 0.1, "max_new_tokens": 512, "repetition_penalty": 1.17},
    }
)

print(data[0]['generated_text'])