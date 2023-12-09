#!/usr/bin/env python3
import argparse
import json
from time import sleep
from prepare import save_interview
from langchain.chat_models import ChatLiteLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def init_model(model, params):
    # integrating liteLLM to provide a standard I/O interface for every LLM
    # see https://docs.litellm.ai/docs/providers for list of supported providers
    model_params = {
            'temperature': params['temperature'],
            'max_tokens': params['max_new_tokens'],
            'top_p': params['top_p'],
            'presence_penalty': params.get('repetition_penalty', 1.0)

    }
    return model_params, ChatLiteLLM(model=model, **model_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interview executor for LangChain')
    parser.add_argument('--input', type=str, required=True, help='path to prepare*.ndjson from prepare stage')
    parser.add_argument('--model', type=str, default='openai/chatgpt', help='model to use')
    parser.add_argument('--params', type=str, required=True, help='parameter file to use')
    parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
    args = parser.parse_args()

    # Load params and init model
    params, model = init_model(args.model, json.load(open(args.params)))

    # Load interview
    interview = [json.loads(line) for line in open(args.input)]
    results = []

    for idx, challenge in enumerate(interview):
        print(f"{idx+1}/{len(interview)} {challenge['name']} {challenge['language']}")
        chain = LLMChain(llm=model, prompt=PromptTemplate(template='{input}', input_variables=['input']))
        answer = chain.run(input=challenge['prompt'])

        print()
        print(answer)
        print()

        result = challenge.copy()
        result['answer'] = answer
        result['params'] = params
        result['model'] = args.model
        result['runtime'] = 'langchain'

        results.append(result)

        if args.delay:
            sleep(args.delay)

    save_interview(args.input, 'none', args.params, args.model, results)