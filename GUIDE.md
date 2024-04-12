# Evaluation Guide

## Prepare Enviroment

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip3 install -r requirements.txt`

## Evaluating Instruct

We will use `MODEL=cognitivecomputations/dolphincoder-starcoder2-7b` as an example, replace with the HF-Hub path of the model you wish to evaluate.

### Build prompts

The first step of an evaluation is to convert an interview into prompts using the `prepare.py` script:

`python3 ./prepare.py --chat $MODEL --interview junior-v2`

The prompts will be written to `results/prepare_junior-v2_python-javascript_chat-simple-cognitivecomputations-dolphincoder-starcoder2-7b.ndjson`

Let's also prepare to run the `senior` interview at the same time:

`python3 ./prepare.py --chat $MODEL --interview junior-v2`

### Run Inference

#### Local GPU (CUDA)

TODO

#### Remote GPU (Modal)

TODO

## Evaluating Code-Completion

TODO

## Evaluating Fill-in-the-Middle

TODO