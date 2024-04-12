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

### Using text-generation-webui

Start text-generation-webui with with the following flags: --listen --api
Here is some [helpful info](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API) on how to set it up if you are having trouble.
Then run the interview command using the flags you need:
- `--input` the file generated from the prepare.py script.
- `--params` the parameters you wish to use for the evaluation (these would most likely overwrite the parameters in ooba, not tested yet, but should be the case)
- `--model` self explanatory
- `--host` when starting ooba from terminal/powershell/cmd it will give you the OpenAI API URL

Example:  
`python3 interview-oobabooga.py --input results/prepare_junior-v2_python-javascript_chat-simple-cognitivecomputations-dolphincoder-starcoder2-7b.ndjson --params params/default.json --model $MODEL --host 0.0.0.0:5000`

## Evaluating Code-Completion

TODO

## Evaluating Fill-in-the-Middle

TODO