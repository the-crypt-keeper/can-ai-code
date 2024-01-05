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

**1/4** Evaluate WizardLM/WizardCoder-33B-V1.1 (FP16), dolphin-2.7-mixtral-8x7b (EXL2 3 and 4 bits), budecosystem/code-millenials-13b and 34b (both at FP16).

**1/2** Evaluate NousResearch/Nous-Hermes-2-SOLAR-10.7B.

**1/1** Evaluate ajibawa-2023/Code family (all good except Python-33B potentially has some issue).

**12/31** Happy new year!  Evaluated TinyLlama 1.1B, it unfortunately fails.  OpenLlama v2 also awful (but it's very old, so that's not unexpected).

**12/29** Evaluate dolphin-2.5-phi, re-evaluate Falcon and WizardCoder-15B via GGUF and remove all old GGML quants. Evaluate replit/replit-code-v1_5-3b.

**12/28** Evaluate Nous Hermes 2 Yi 34B quants, great model but recommend to avoid the 3bit EXL2.

**12/28** Split up mixtral and mistral, evaluate a wide array of mixtral instruct quants.

## Test Suites

`junior-v2` is a multi-language (Python, JavaScript) suite of 12 tests created for this project to test small LLM coding performance.  This project provides all necessary components to execute this evaluation.

:construction: `humaneval` is a Python-only suite of 164 tests created by OpenAI.  This project provides template scripts to prepare and execute the humaneval interview, as well as result extraction scripts to help their evaluator. See https://github.com/openai/human-eval for more information.

## [Click to see Leaderboard on HF Spaces](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results)

## [Click to see Comparisons on HF Spaces](https://huggingface.co/spaces/mike-ravkine/can-ai-code-compare)

### Results data

All model answers and evaluation results are now included inside this repository!  Install a recent release of streamlit `pip install streamlit==1.23` then `streamlit run app.py` or `streamlit run compare-app.py` to run the above webapps locally.

### Results HumanEval

:construction: [humaneval/](humaneval/) development work is currently paused, there's other projects that are much further along.

See https://github.com/my-other-github-account/llm-humaneval-benchmarks and https://github.com/abacaj/code-eval for large lists of Humaneval LLM benchmark results.

## Repository Structure

### Interviews

* `junior-v2/*.yaml` - junior coder interview questions (stable)
* `senior/*.yaml` - senior coder interview questions (WIP)

### Prepare

* `prompts/*.txt` - LLM prompt templates for the various models
* `prepare.py` - Applies templates to question turning them into language- and model-specific prompts suitable for interview

#### Prompts

See [prompts/](prompts/) for all prompts references in the leaderboard.

### Interview

* `params/*.json` - Sampling hyper-parameter sets (used by all interview scripts)
* `interview-*.py` - Interview scripts

#### Parameters

See [params/](params/) for all params references in the leaderboard.

### Evaluate

* `evaluate.py` - Run tests for the generated code in a sandbox and grades each answer
* `app.py` - Streamlit webapp to explore results, see https://huggingface.co/spaces/mike-ravkine/can-ai-code-results

### Compare

* `compare.py` - Performs comparisons between evaluations, optionally calling out to an LLM for analysis
* `compare-app.py` - Streamlit webapp to explore comparisons, see https://huggingface.co/spaces/mike-ravkine/can-ai-code-compare
* `compare/*.yaml` - Compare configurations
* `compare/*.json` - Compare results

## Interviewers: API

| API Runtime              | Script         |
|--------------------------|----------------|
| LiteLLM (OpenAI, etc..)  | `interview-litellm.py` |
| OobaBooga/KoboldCpp      | `interview-oobabooga.py` |
| Huggingface Inference    | `interview-hfinference.py` |
| Gradio (HF Spaces)       | `interview-gradio.py` |

## Interviewers: CUDA (Local)

| Quantization Type        | Script                  | Dependency              |
|--------------------------|-------------------------|-------------------------|
| GGUF                     | `interview-llamacpp.py` | llamacpp or ggml binary |
| GPTQ (AutoGptQ)          | `interview-cuda.py`     | auto-gptq==0.5.1        |
| GPTQ (ExLlama)           | `interview-cuda.py`     | exllama @ 3b013cd53c7d413cf99ca04c7c28dd5c95117c0d |
| EXL2, GPTQ (ExLlama2)    | `interview-cuda.py`     | exllamav2 @ 3cabfb0d0672c18ffa1aba9bcae3328cfd86dfe7 |
| HQQ                      | `interview-cuda.py`     | hqq @ 0.1.1             |
| AWQ, FP16 (vLLM)         | `interview-cuda.py`     | vllm==0.2.6             |
| CTranslate2              | `interview-cuda.py`     | ctranslate2>=3.16.0     |
| bitsandbytes             | `interview-cuda.py`     | bitsandbytes==0.41.3    |
| FP16 (Transformers)      | `interview-cuda.py`     | transformers==4.36.1    |

### Running on Modal

The recommended modal wrapper is `interview_modal_cuda11.py` which builds a CUDA11.8 based container with all the above dependencies working. An `interview_modal_cuda12.py` is also provided, but AutoGPTQ and CTranslate2 are not compatible.

Unfortunately the nature of Modal does not allow command-line selection of eitehr LLM model or runtime engine.

To select models, open the script and uncomment the `.run_function(download...)` line of choice.  Note that only one model can be selected at a time.   To add a new model, implement a new `download...` function.

To select runtime, open the script and uncomment one of the `RUNTIME` options. Note that for `transformers` you must also specify `QUANT`.

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

* [Development of a Senior coder test suite](https://github.com/the-crypt-keeper/can-ai-code/issues/141)
* Open [Model Request](https://github.com/the-crypt-keeper/can-ai-code/labels/model%20request) issues
