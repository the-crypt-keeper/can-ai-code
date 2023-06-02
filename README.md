# Can AI Code?

A self-evaluating interview for AI coding models.

NOTE: This branch is a work in progress refactor of both the test suite and executor.  See https://github.com/the-crypt-keeper/can-ai-code/pull/9 for discussion.

## Key Ideas

* Interview questions written by humans, test taken by AI
* Sandbox enviroment (Docker-based) for untrusted Python and NodeJS execution
* Compare LLM models against each other
* For a given LLM, compare prompting techniques and hyper-parameters

## Results

:construction: :construction: V2 - UNDER CONSTRUCTION :construction: :construction: 

| Model | Quant | Size | License | Prompt | Parameters | Python | JavaScript |
|-------|--------------|------|---------|--------|------------|--------|------------|
| openai/gpt-3.5-turbo      | API   | 170B | Closed | openai-chatgpt         | precise | 65/65 :1st_place_medal: | 65/65 :1st_place_medal: |
| ai21/j2-jumbo-instruct    | API   | 178B | Closed | ai21-j2-jumbo-instruct | precise | 55/65                   | 54/65                   |
| cohere/command-nightly    | API   | 52B  | Closed | cohere-command-nightly | precise | 52/65                   | 49/65                   |
| bigcode/tiny_starcoder_py | FP32  | 159M | Open   | starcoder-fim          | precise | 38/65                   | 0/0                     |
| bigcode/starcoder         | FP32  | 16B  | Open   | starcoder-fim          | precise | 46/65                   | 45/65                   |
| VicUnlocked-30B-LoRA      | GPTQ 4b/128g | 30B | Open | Vicuna-1p1         | precise | 49/65                   | 48/65                   |
| Manticore-13B             | ggmlv3 q5_0  | 13B | Open | Wizard-Vicuna      | precise | 42/65                   | 40/65                   |
| Manticore-13B             | ggmlv3 q5_0  | 13B | Open | Manticore          | precise | 36/65                   | 41/65                   |
| Vicuna-1.1-7B             | ggmlv3 q5_0  |  7B | Open | Vicuna-1p1         | precise | 44/65                   | 41/65                   |

## Data Sets

:new: Model answers are now included inside this repository!  See `results/`

## Repository Structure

### Prepare

* `junior-dev/*.yaml` - Interview questions (multi-language)
* `prompts/*.txt` - System prompts for the various models
* `prepare.py` - Applies templates to question turning them into language- and model-specific prompts suitable for interview

### Interview

* `interview-langchain.py` - Run using LangChain model interface
* `interview-oobabooga.py` - Run using OoobaBooga (or KoboldCpp) remote API
* `interview-gptq-modal.py` - Run GPTQ on Modal remote GPU rental platform
* `intreview-llamacpp.py` - Run GGML llama.cpp model on local (or remote via ssh) CPU/GPU

* `interview-hfinference.py` - Run Huggingface Inference API to run various models
* `interview-starchat.py` - Run Huggingface Space to run Starchat model **not updated for v2 yet**
* `interview-starcoder.py` - Use Huggingface Transformers to run Starcoder models on local GPU

### Evaluate

* `evaluate.py` - Run tests for the generated code in a sandbox and grades each answer

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

TODO update for v2

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

Contributions are welcome!  Especially looking for additional interview sets and improvements to questions - open a PR! 

* Evaluate more 30B and 65B open langauge models
* If the models are offered error messages or failing test results, could they produce better code?
* Can tweaking prompts improve performance?
