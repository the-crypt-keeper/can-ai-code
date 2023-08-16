#!/usr/bin/env python3
from langchain import LLMChain, PromptTemplate
import argparse
import json
from time import sleep
from prepare import save_interview
from langchain.llms import ChatLiteLLM

def init_model(model, params):
    # LangChain did not bother to standardize the names of any of the parameters,
    # or even how to interact with them.  This is a hack to make things consistent.

    # integrating liteLLM to provide a standard I/O interface for every LLM
    model_params = {
            'temperature': params['temperature'],
            'max_tokens': params['max_new_tokens'],
            'top_p': params['top_p'],
            'presence_penalty': params['repetition_penalty']

    }
    if model == 'ai21/j2-jumbo-instruct':
        model_name = "j2-jumbo-instruct"
    elif model == 'openai/chatgpt':
        model_name = "gpt-3.5-turbo"
    elif model == 'openai/gpt4':
        model_name = "gpt-4"
    elif model == 'cohere/command-nightly':
        model_name = "command-nightly"
    # [TODO] Add replicate, hugging face, VertexAI, see https://docs.litellm.ai/docs/completion/supported
    else:
        raise Exception('Unsupported model/provider')
    return model_params, ChatLiteLLM(model=model_name, **model_params)

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