# Can AI Code?

A self-evaluating interview for AI coding models.

## Key Ideas

* Interview questions written by humans, test taken by AI
* Sandbox enviroment (Docker-based) for untrusted Python and NodeJS execution
* Compare LLM models against each other
* For a given LLM, compare prompting techniques and hyper-parameters

## Supported Test Suites

`junior-dev` is a multi-language (Python, JavaScript) suite of 12 tests created for this project to test small LLM coding performance.  This project provides all necessary components to execute this evaluation.

`humaneval` is a Python-only suite of 164 tests created by OpenAI.  This project provides template scripts to prepare and execute the humaneval interview, as well as result extraction scripts to help their evaluator. See https://github.com/openai/human-eval for more information.

## Results junior-dev

| Model | Quant | Size | License | Prompt | Params | Python | JavaScript |
|-------|--------------|------|---------|--------|------------|--------|------------|
| openai/gpt-3.5-turbo      | API   | 170B | Closed | openai-chatgpt         | precise | 65/65 :1st_place_medal: | 65/65 :1st_place_medal: |
| ai21/j2-jumbo-instruct    | API   | 178B | Closed | ai21-j2-jumbo-instruct | precise | 55/65 :3rd_place_medal: | 54/65                   |
| cohere/command-nightly    | API   | 52B  | Closed | cohere-command-nightly | precise | 52/65                   | 49/65                   |
| bigcode/tiny_starcoder_py | FP32  | 159M | Open   | starcoder-fim          | precise | 38/65                   | 0/0                     |
| bigcode/starcoder         | FP32  | 16B  | Open   | starcoder-fim          | precise | 46/65                   | 45/65                   |
| bigcode/starchat          | FP32  | 16B  | Open   | starchat               | precise | 48/65                   | 53/65 :3rd_place_medal: |
| VicUnlocked-30B-LoRA      | GPTQ 4b/128g | 30B | Open | Vicuna-1p1         | precise | 49/65                   | 48/65                   |
| Manticore-13B             | ggmlv3 q5_0  | 13B | Open | Wizard-Vicuna      | precise | 42/65                   | 40/65                   |
| Manticore-13B             | ggmlv3 q5_0  | 13B | Open | Manticore          | precise | 36/65                   | 41/65                   |
| Manticore-13B-Chat-Pyg-Guanaco | ggmlv3 q5_0  | 13B | Open |  Manticore-YearZero | precise | 43/65             | 50/65                   |
| Manticore-13B-Chat-Pyg-Guanaco | ggmlv3 q5_0  | 13B | Open |  Manticore-YearZero | mirostat | 43/65            | 50/65                   |
| Vicuna-1.1-7B             | ggmlv3 q5_0  |  7B | Open | Vicuna-1p1         | precise | 44/65                   | 41/65                   |
| Vicuna-1.1-13B            | ggmlv3 q5_0  | 13B | Open | Vicuna-1p1         | precise | 57/65 :2nd_place_medal: | 57/65 :2nd_place_medal: |
| WizardLM-7B-uncensored    | ggmlv3 q5_1  |  7B | Open | Wizard-Vicuna      | precise | 51/65                   | 37/65                   |
| WizardLM-13B-1.0          | ggmlv3 q5_0  | 13B | Open | Wizard-Vicuna      | precise | 51/65                   | 50/65                   |
| Wizard-Vicuna-13B-Uncensored | ggmlv3 q5_0 | 13B | Open | Wizard-Vicuna      | precise | 31/65 :poop:          | 48/65                   |
| Guanaco-7B                | ggmlv3 q5_0  |  7B | Open | Guanaco            | precise | 41/65                   | 41/65                   |
| Guanaco-13B               | ggmlv3 q5_0  | 13B | Open | Guanaco            | precise | 29/65 :poop:            | 39/65                   |

:new: Model answers are now included inside this repository!  See `results/`

## Results HumanEval

:construction: HumanEval work is under active development.

| Model |     Quant    | Size | License | Prompt |    Params  | Python |
|-------|--------------|------|---------|--------|------------|--------|
| VicUnlocked-30B-LoRA      | GPTQ 4b/128g | 30B | Open | Vicuna-1p1         | precise | 20/164 |

## Repository Structure

### Prepare: junior-dev

* `junior-dev/*.yaml` - Interview questions (multi-language)
* `prompts/*.txt` - System prompts for the various models
* `prepare.py` - Applies templates to question turning them into language- and model-specific prompts suitable for interview

### Prepare: humaneval

* `humaneval.py` - Applies templates to humaneval interview suite turning them into prompts suitable for interview

### Interview: Common

`model_parameters/*.json` - Sampling hyper-parameter sets (used by all interview scripts)

### Interview: Langchain 

`interview-langchain.py` provides a LangChain interview executor.

To add a new model, look at `init_model`.

### Interview: OobaBooga/KoboldCpp API

`interview-oobabooga.py` provides a text-generation-ui/koboldcpp API compatible interview executor.

### Interview: GPTQ

`interview-gptq-modal.py` - Run Ooba-Booga fork of GPTQ on Modal

`interview-autogptq-modal.py` - Run latest AutoGPTQ on Modal

### Interview: Llama.cpp (GGML)

`Interview-llamacpp.py` provides an executor to wrap `main` on local (or remote via ssh) CPU/GPU

### Interview: Huggingface APIs

* `interview-hfinference.py` - Use Huggingface Inference API to run various models
* `interview-starchat.py` - Use Huggingface Spaces to run Starchat model
* `interview-starcoder.py` - Use Huggingface Transformers to run Starcoder models on local GPU

### Evaluate: junior-dev

`evaluate.py` - Run tests for the generated code in a sandbox and grades each answer

## Evaluate: humaneval

Use `humaneval.py --answers` to convert any interview output into .jsonl format and use the upsteam evaluator: https://github.com/openai/human-eval

## Question Format

A set of interview questions is a folder of .yaml files.  Each Question is a top-level key:

```yaml
SanityList:
    Signature: "things()"
    Input: "with no inputs"
    Output: "a list with three values: the number 5, the string 'foobar', the capital city of Spain"
    Fact: "the capital city of Spain is Madrid"
    Description: "List function, see if the model can combine input facts with internal knowledge."
    Checks:
        input_name:
            assert: "f.name"
            eq: "things"
```

In this example `SanityList` is the name of the interview question.

The first four fields are used by `prepare.py` to create the interview:

- `Signature` is the desired function signature
- `Input` describes the function inputs
- `Output` describes the function outputs
- `Fact` is optional and provides any context that is required to correctly perform the task

These 4 variables along with `language` (either `python` or `javascript`) are used to expand templates in `prompts/`.

The last two fields are used by `evaluate.py` to judge the results:

- `Description` is a human-readable explanation of why this test is useful
- `Checks` defines the expected behavior of the output.

### Checks and the 'f' object

Each check has a name, some `assert` value (python code) and an expected `eq` value.

The f object represents the sandbox view of the function.  Static analysis is performed on the function signature to extract the `f.name` and `f.args` fields, while `f.call` allows for function evaluation.

## Using this Repository

TODO

### Prompts

`Vicuna-1p1.txt`

`starcoder-fim*.txt`

`Manticore-YearZero.txt` (from https://www.reddit.com/r/LocalLLaMA/comments/13yfask/manticore13bchatpygguanacoggmlq4_0_americas_next/)

### Parameters

`precise.json`

`mirostat.json` (from https://www.reddit.com/r/LocalLLaMA/comments/13yfask/manticore13bchatpygguanacoggmlq4_0_americas_next/)

## Output formats

All scripts output automatically named .ndjson files to the `results/` directory.

Each stage outputs a super-set of fields from the stage before it, so its possible to feed eval/interview back to interview (to re-run the questions) or back to eval (to re-run the eval).

### prepare

`results/prepare_{interview}_{languages}_{template}.ndjson`

Fields:

- all Question fields (Signature, Input, Output, Fact, Description)
- name
- language
- prompt

### interview

`results/interview_{interview}_{languages}_{template}_{templateout}_{params}_{model}_{timestamp}.ndjson`

Fields:
- all `prepare` fields
- model
- params
- answer

### eval

`results/eval_{interview}_{languages}_{template}_{templateout}_{params}_{model}_{timestamp}.ndjson`

Fields:
- all `eval` fields
- status
- passed
- total
- checks

# Roadmap / Future Work

## Interesting Models

* Evaluate Llama and Alpaca 65B open models
* Evaluate codet5p: https://huggingface.co/Salesforce/codet5p-16b
* Evaluate CodeAlpaca: https://github.com/sahil280114/codealpaca

## Additional Interviews

* Port HumanEval, a standard LLM code benchmark with 164 tests: https://github.com/openai/human-eval

## Investigations

* If the models are offered error messages or failing test results, could they produce better code?
* Can tweaking prompts improve performance?
