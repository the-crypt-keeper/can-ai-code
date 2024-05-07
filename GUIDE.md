# Evaluation Guide

## Prepare Enviroment

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip3 install -r requirements.txt`

## Evaluating Instruct

We will use `MODEL=google/codegemma-7b-it` as an example, replace with the HF-Hub path of the model you wish to evaluate.

### Build prompts

The first step of an evaluation is to convert an interview into prompts using the `prepare.py` script:

`python3 ./prepare.py --chat $MODEL --interview junior-v2`

The prompts will be written to `results/prepare_junior-v2_python-javascript_chat-simple-google-codegemma-7b-it.ndjson`

Let's also prepare to run the `senior` interview at the same time:

`python3 ./prepare.py --chat $MODEL --interview junior-v2`

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