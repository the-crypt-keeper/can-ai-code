# Evaluation Guide

## Prepare Enviroment

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Common Options

All interview scripts accept the following common options:

* `--interview` directly run instruct-completion (default: senior)
* `--input` run a pre-prepared interview used for completion and fim

## API Evaluations (Instruct)

### Evaluate a Remote API (LiteLLM)

`python ./interview_litellm.py --model <provider>/<model_id> --apikey <key>`

See LiteLLM documentation for the full list of supported providers.

### Evaluate a Local/Self-Hosted API (LiteLLM)

`python ./interview_litellm.py --model openai/<model_id> --apibase http://<host>:<port>/`

If the runtime cannot be inferred from the endpoint, you will be asked to provide `--runtime`

#### Evaluate a Local ollama API

`ollama serve <model_id>`

`python ./interview_litellm.py --model ollama_chat/<model_id>`

#### Evaluate a Local llama-server API

`llama-server -m /home/mike/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf -c 8192 -fa -ngl 99 --host 0.0.0.0 --port 8080`

Note: ` -fa` enables flash attention, ` -ngl 99` enables GPU offloading

`python3 ./interview-litellm.py --model openai/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf --apibase http://127.0.0.1:8080`

#### Evaluate a Local koboldcpp API

`koboldcpp /home/mike/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf --contextsize 8192 --flashattention --gpulayers 99 --usecublas 1 --host 0.0.0.0 --port 8080`

Note: `--flashattention` enables flash attention, `--gpulayers 99 --usecublas 1` enables GPU offloading

`python3 ./interview-litellm.py --model openai/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf --apibase http://127.0.0.1:8080`

## Model Evaluations (Instruct)

### Evaluate a model with local GPU (CUDA)

The local CUDA executor will use all available GPUs by default, use `CUDA_VISIBLE_DEVICES` if you have connected accelerators you don't want used.

#### Backend: Transformers

`pip install -r requirements.txt -r requirements-transformers.txt`

`python ./interview_cuda.py --model <model> --runtime transformers`

#### Backend: vLLM

`pip install -r requirements.txt -r requirements-vllm.txt`

`python ./interview_cuda.py --model <model> --runtime vllm`

#### Backend: Exllamav2

`pip install wheel && pip install -r requirements.txt -r requirements-exl2.txt`

`python ./interview_cuda.py --model <model> --runtime exllama2`

### Evaluate a model with remote GPU (Modal)

`python ./interview_modal.py --model <model> --runtime <runtime> --gpu <gpu>`

See modal docs for valid GPUs.

## Model Evaluations (Code Completion)

TODO

## Model Evaluations (Fill-in-the-Middle)

TODO

## Run Self-Checks and View Results

`bulk-eval.sh` is a quick and easy way to run the `evaluate.py` script for all `results/interview*` it finds.

`streamlit run app.py "results/eval*"` will then show you local results only.
