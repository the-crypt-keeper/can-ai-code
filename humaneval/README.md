## HumanEval Integration



### Prepare

Use `humaneval.py --template` to apply templates to humaneval interview suite turning them into prompts suitable for interview.

See `humaneval/Vicuna-1p1-HumanEval.txt` for an example template, the format is different then internal tests.

### Evaluate

Use `humaneval.py --answers` to convert any interview output into .jsonl format and use the upsteam evaluator: https://github.com/openai/human-eval
