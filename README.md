# Can AI Code?

A self-evaluating interview for AI coding models.

## Key Ideas

* Interview questions written by humans, test taken by AI
* Sandbox enviroment (Docker-based) for untrusted Python and NodeJS execution
* Compare LLM models against each other
* For a given LLM, compare prompting techniques and hyper-parameters

## Supported Test Suites

`junior-dev` is a multi-language (Python, JavaScript) suite of 12 tests created for this project to test small LLM coding performance.  This project provides all necessary components to execute this evaluation.

:construction: `humaneval` is a Python-only suite of 164 tests created by OpenAI.  This project provides template scripts to prepare and execute the humaneval interview, as well as result extraction scripts to help their evaluator. See https://github.com/openai/human-eval for more information.

## Results junior-dev

:new: The leaderboard table got too difficult to read, explore the interactive leaderboard: https://huggingface.co/spaces/mike-ravkine/can-ai-code-results :new:

All model answers and evaluation results are now included inside this repository!  See `results/`

## Results HumanEval

:construction: HumanEval work is under active development.

| Model |     Quant    | Size | License | Prompt |    Params  | Python |
|-------|--------------|------|---------|--------|------------|--------|
| VicUnlocked-30B-LoRA      | GPTQ 4b/128g | 30B | Open | Vicuna-1p1         | precise | 20/164 |

## Repository Structure

The repository is logically grouped into three parts: prepare, interview, evaluate.

### Prepare

#### junior-dev

* `junior-dev/*.yaml` - Interview questions (multi-language)
* `prompts/*.txt` - System prompts for the various models
* `prepare.py` - Applies templates to question turning them into language- and model-specific prompts suitable for interview

#### humaneal

See [humaneval/](humaneval/).

### Interview

`model_parameters/*.json` - Sampling hyper-parameter sets (used by all interview scripts)

#### LangChain 

`interview-langchain.py` provides a LangChain interview executor.

To add a new model, update `init_model` to add parameter mappings and adapter instance.

#### OobaBooga/KoboldCpp API

`interview-oobabooga.py` provides a text-generation-ui/koboldcpp API compatible interview executor.

#### GPTQ

`interview-gptq-modal.py` - Run Ooba-Booga fork of GPTQ on Modal

`interview-autogptq-modal.py` - Run latest AutoGPTQ on Modal

#### Llama.cpp (GGML)

`Interview-llamacpp.py` provides an executor to wrap `main` on local (or remote via ssh) CPU/GPU

#### Huggingface APIs

* `interview-hfinference.py` - Use Huggingface Inference API to run various models
* `interview-starchat.py` - Use Huggingface Spaces to run Starchat model
* `interview-starcoder.py` - Use Huggingface Transformers to run Starcoder models on local GPU

### Evaluate

#### junior-dev

`evaluate.py` - Run tests for the generated code in a sandbox and grades each answer

#### humaneval

See [humaneval/](humaneval/).

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
