# Can AI Code?

A self-evaluating interview for AI coding models.

## Key Ideas

* Interview questions written by humans, test taken by AI
* Sandbox enviroment (Docker-based) for untrusted Python and NodeJS execution
* Compare LLM models against each other
* For a given LLM, compare prompting techniques and hyper-parameters

## Results

TODO: Update for v2

## Results (chart)

TODO: Update for v2

## Data Sets

TODO: Update for v2

## Repository Structure

* `junior-dev/*.yaml` - Interview questions (multi-language)
* `prompts/*.txt` - System prompts for the various models
* `prepare.py` - Specializes question into prompts for a specific language
* `interview-langchain.py` - Run using LangChain model interface
* `interview-oobabooga.py` - Run using OobbaBooga remote API model interface
* `interview-starchat.py` - Run Huggingface Space to run Starchat model
* `interview-starcoder.py` - Run Huggingface Transformers to run Starcoder models on local GPU
* `interview-gptq-modal.py` - Run GPTQ on Modal remote GPU rental platform
* `intreview-llamacpp.sh` - Run GGML llama.cpp model on local CPU/GPU
* `evaluate.py` - Run tests for the generated code in a sandbox and grades each answer
* `report.py` - (WIP - not yet complete) Compare results from multiple interviews

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

`Signature` is the desired function signature
`Input` describes the function inputs
`Output` describes the function outputs
`Fact` is optional and provides any context that is required to correctly perform the task

These 4 variables along with `language` (either `python` or `javascript`) are used to expand templates in `prompts/`.

The last two fields are used by `evaluate.py` to judge the results:

`Description` is a human-readable explanation of why this test is useful
`Checks` defines the expected behavior of the output.

### Checks and the 'f' object

Each check has a name, some `assert` value (python code) and an expected `eq` value.

The f object represents the sandbox view of the function.  Static analysis is performed on the function signature to extract the `f.name` and `f.args` fields, while `f.call` allows for function evaluation.

## Using this Repository

TODO update for v2

## Interview format

TODO update for v2

# Roadmap / Future Work

Contributions are welcome!  Especially looking for additional interview sets and improvements to questions - open a PR! 

* Evaluate 30B and 65B open langauge models
* If the models are offered error messages or failing test results, could they produce better code?
* Can tweaking prompts improve performance?
