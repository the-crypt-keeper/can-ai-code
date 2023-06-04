## HumanEval Integration

:construction: Work in progress.

### Prepare

Use `humaneval.py --template` to apply templates to humaneval interview suite turning them into prompts suitable for interview.

#### Templates

Note the format here is different, uses a single {{prompt}} variable as the input.

`Vicuna-1p1-HumanEval.txt`

`Wizard-7B-HumanEval.txt`

`Llama-Humaneval.txt`

### Evaluate

Use `humaneval.py --answers` to convert any interview output into .jsonl format and use the upsteam evaluator: https://github.com/openai/human-eval
