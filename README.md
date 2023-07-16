<div style="text-align: center;">

# Can AI Code?

![A cute robot working on a laptop](img/can-ai-code-small.png "A cute robot working on a laptop")

A self-evaluating interview for AI coding models.

</div>

## Key Ideas

* Interview questions written by humans, test taken by AI
* Sandbox enviroment (Docker-based) for untrusted Python and NodeJS execution
* Provide reference coding prompts tuned for each LLM
* Compare LLM models coding performance against each other
* Evaluate effects of prompting techniques and sampling parameters as well as the impact of quantization methods on LLM coding performance

## News

**7/16** Airboros-1.4 evaluation and comparison between 1.4 and 1.4.1 has been added.  Note [the code extractor is failing on some of the GPTQ answers](https://github.com/the-crypt-keeper/can-ai-code/issues/43) so the GPTQ quants are scoring lower then they should be.

**7/16** Updated results for Vicuna-1.3 AWQ and merged the `compare` feature!  Browse [Comparisons](https://huggingface.co/spaces/mike-ravkine/can-ai-code-compare) at our new space.

**7/15** bitsandbytes [INT8](https://github.com/TimDettmers/bitsandbytes) and [NF4](https://huggingface.co/blog/4bit-transformers-bitsandbytes) now supported via `interview-transformers-modal.py`

**7/15** fixed the input() bug in the python evaluator and re-scored affected models.

**7/15** New `interview-gradio.py` replaces several older gradio-based interviewers, refactor the interview list in the README.

**7/14** Completed evaluations on Falcon (7B and 40B) and Vicuna-1.3 (7B, 13B, 33B) across a variety of quantizations, see [can-ai-code Leaderboard](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results) for the results: select a language, then unselect "Show Best Result from each model" to see per-quant results.

**7/12** [AWQ](https://github.com/mit-han-lab/llm-awq) now supported via `interview-awq-model.py` but only 4-bit for now as the authors haven't released 3-bit code.

## Test Suites

`junior-dev` is a multi-language (Python, JavaScript) suite of 12 tests created for this project to test small LLM coding performance.  This project provides all necessary components to execute this evaluation.

:construction: `humaneval` is a Python-only suite of 164 tests created by OpenAI.  This project provides template scripts to prepare and execute the humaneval interview, as well as result extraction scripts to help their evaluator. See https://github.com/openai/human-eval for more information.

## [Click to see Leaderboard on HF Spaces](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results)

## [Click to see Comparisons on HF Spaces](https://huggingface.co/spaces/mike-ravkine/can-ai-code-compare)

### Results data

All model answers and evaluation results are now included inside this repository!  Install a recent release of streamlit `pip install streamlit==1.23` then `streamlit run app.py` or `streamlit run compare-app.py` to run the above webapps locally.

### Results HumanEval

:construction: [humaneval/](humaneval/) development work is currently paused, there's other projects that are much further along.

See https://github.com/my-other-github-account/llm-humaneval-benchmarks and https://github.com/abacaj/code-eval for large lists of Humaneval LLM benchmark results.

## Repository Structure

### Prepare

* `junior-dev/*.yaml` - Interview questions (multi-language)
* `prompts/*.txt` - LLM prompt templates for the various models
* `prepare.py` - Applies templates to question turning them into language- and model-specific prompts suitable for interview

#### Prompts

(WIP)

`Vicuna-1p1.txt`

`starcoder-fim*.txt`

`Manticore-YearZero.txt` (from https://www.reddit.com/r/LocalLLaMA/comments/13yfask/manticore13bchatpygguanacoggmlq4_0_americas_next/)

### Interview

* `params/*.json` - Sampling hyper-parameter sets (used by all interview scripts)
* `interview-*.py` - Interview scripts

#### Parameters

(WIP)

`precise.json`

`mirostat.json` (from https://www.reddit.com/r/LocalLLaMA/comments/13yfask/manticore13bchatpygguanacoggmlq4_0_americas_next/)

### Evaluate

* `evaluate.py` - Run tests for the generated code in a sandbox and grades each answer
* `app.py` - Streamlit webapp to explore results, see https://huggingface.co/spaces/mike-ravkine/can-ai-code-results

### Compare

* `compare.py` - Performs comparisons between evaluations, optionally calling out to an LLM for analysis
* `compare-app.py` - Streamlit webapp to explore comparisons, see https://huggingface.co/spaces/mike-ravkine/can-ai-code-compare
* `compare/*.yaml` - Compare configurations
* `compare/*.json` - Compare results

## Interviewers

|        Script            |     Runtime    | Models | Quants | Local/Remote |
|--------------------------|----------------|--------|--------|--------------|
| `interview-langchain.py` | langchain      | lots   | n/a    | n/a          |
| `interview-oobabooga.py` | oobabooga, koboldcpp | lots | yes | remote      |
| `interview-autogptq.py`  | autogptq       | lots   | gptq   | local + modal via `interview-autogptq-modal.py` |
| `interview-transformers.py` | transformers | lots | yes | local + modal via `interview-transformers-modal.py` |
| `interview-exllama-modal.py` | exllama | llama | gptq | remote via modal |
| `interview-vllm-modal.py` | vllm | llama | n/a | remote via modal |
| `interview-awq-modal.py` | awq | llama, falcon | awq | remote via modal |
| `interview-llamacpp.py`  | ggml, ggllm, llamacpp | lots | GGML | local + remote via ssh |
| `interview-hfinference.py` | hf-inference-api | lots | n/a | remote |
| `interview-gradio.py`    | gradio | lots | n/a | remote |

### Notes on adding new models

* LangChain: To add a new model, update `init_model` to add parameter mappings and adapter instance.

* All modal scripts:   The nature of Modal does not allow command-line selection of LLM model.  In order to select models, you'll have to open the script and uncomment the `.run_function(download...)` line of choice.  Note that only one model can be selected at a time.   To add a new model, implement a new `download...` function.  Quantization parameters are only required if the model does not contain a `quantize_config.json`.

### Notes on llamacpp

For llama (https://github.com/ggerganov/llama.cpp): --main main --args=""

For starcoder (https://github.com/ggerganov/ggml): --main starcoder --args=""

For falcon (https://github.com/cmp-nct/ggllm.cpp): --main falcon_main --args="--no-penalize-nl"

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
- runtime

### eval

`results/eval_{interview}_{languages}_{template}_{templateout}_{params}_{model}_{timestamp}.ndjson`

Fields:
- all `eval` fields
- status
- passed
- total
- checks

# Roadmap / Future Work

* See all open [Model Request](https://github.com/the-crypt-keeper/can-ai-code/labels/model%20request) issues
* If the models are offered error messages or failing test results, could they produce better code?
* [Can tweaking prompts improve performance?](https://github.com/the-crypt-keeper/can-ai-code/issues/37)
