# Evaluation Guide

## Prepare Enviroment

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip3 install -r requirements.txt`

## Evaluting Instruct with local API

### Prepare "chat-simple" template

```
python3 ./prepare.py --template prompts/chat-simple.txt
```

### Launch API and evaluate

If successful this step will drop two interview*.json files (one for senior and another for junior-v2) into the results/ directory.

#### llama-server

```
llama-server -m /home/mike/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf -c 8192 -fa -ngl 99 --host 0.0.0.0 --port 8080
```

`-fa` enables flash attention, `-ngl 99` enables GPU offloading

```
python3 ./interview-litellm.py --runtime llama --apikey xx --apibase http://127.0.0.1:8080 --params params/greedy-openai.json --input results/prepare_senior_python-javascript_chat-simple.ndjson,results/prepare_junior-v2_python-javascript_chat-simple.ndjson
```

#### koboldcpp

```
koboldcpp /home/mike/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf --contextsize 8192 --flashattention --gpulayers 99 --usecublas 1 --host 0.0.0.0 --port 8080
```

`--flashattention` enables flash attention, `--gpulayers 99 --usecublas 1` enables GPU offloading

```
python3 ./interview-litellm.py --runtime koboldcpp --apikey xx --apibase http://127.0.0.1:8080 --params params/greedy-openai.json --input results/prepare_senior_python-javascript_chat-simple.ndjson,results/prepare_junior-v2_python-javascript_chat-simple.ndjson
```

#### ollama

```
python3 ./interview-litellm.py --model 'ollama_chat/<model>' --params params/greedy-openai.json --input results/prepare_senior_python-javascript_chat-simple.ndjson,results/prepare_junior-v2_python-javascript_chat-simple.ndjson
```

## Evaluating Instruct with CUDA

We will use `MODEL=google/codegemma-7b-it` as an example, replace with the HF-Hub path of the model you wish to evaluate.

### Build chat-formatted prompts

The first step of an evaluation is to convert an interview into prompts using the `prepare.py` script:

`python3 ./prepare.py --chat $MODEL`

The prompts will be written to `results/prepare_junior-v2_python-javascript_chat-simple-google-codegemma-7b-it.ndjson` and `results/prepare_senior_python-javascript_chat-simple-google-codegemma-7b-it.ndjson`

### Run Inference

#### Local GPU: Transformers

`pip3 install -r requirements.transformers.txt`

```
python3 ./interview_cuda.py \
         --runtime transformers \
         --model_name google/codegemma-7b-it \
         --input results/prepare_junior-v2_python-javascript_chat-simple-google-codegemma-7b-it.ndjson \
         --params params/greedy-hf.json
```

#### Local GPU: vLLM

TODO

#### Local GPU: exllamav2

TODO

#### Remote GPU (Modal)

TODO

## Evaluating Code-Completion

TODO

## Evaluating Fill-in-the-Middle

TODO