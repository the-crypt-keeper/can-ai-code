# Can AI Code?

A self-evaluating interview for AI coding models.

## Key Ideas

* Interview questions written by humans, test taken by AI
* Sandbox enviroment (Docker-based) for untrusted Python and NodeJS execution
* Compare LLM models against each other
* For a given LLM, compare prompting techniques and hyper-parameters

## Results

|Model|Notes|Python|JavaScript|
|-----|-----|-----|-----|
|openai/gpt-3.5-turbo|Proprietary 170B|65/65 :1st_place_medal:|62/65 :1st_place_medal:|
|ai21/j2-jumbo-instruct|Proprietary 178B|55/65 :2nd_place_medal:|39/65|
|cohere/command-nightly|Proprietary 52B|48/65|45/65|
|Wizard-Vicuna-13B-Uncensored|Open 13B ggmlv3.q5_0|31/65|48/65 :3rd_place_medal:|
|vicuna-7B-1.1|Open 7B ggmlv3 q5_0|51/65|40/65|
|Manticore-13B|Open 13B ggmlv3.q5_0|47/65|37/65|
|Guanaco-13B|Open 13B GPTQ 4bit|41/65|37/65|
|WizardLM-13B-1.0|Open 13B ggmlv3.q5_0|53/65|**52/65** :2nd_place_medal:|
|WizardLM-7B-Uncensored|Open 7B ggmlv3.q5_1|**54/65** :3rd_place_medal:|37/65|
|VicUnlocked-30B-LoRA|Open 30B GPTQ 4bit|45/65|41/65|
|bigcode/Starchat|Open 15.5B|40/65|45/65|
|bigcode/tiny_starcoder_py|Open 159M|39/65 [1] :construction:|N/A|
|bigcode/Starcoder|Open 15.5B|45/65 [1] :construction:|:construction:|

[1] Starcoder models are prompted with FIM techniques, so these results are not directly comparable to the other chat models. See https://github.com/the-crypt-keeper/tiny_starcoder/blob/can-ai-code/interview-tinystarcoder.py

Evaluation of 65B models is on the Roadmap.  Can you help?  Reach out!

## Results (chart)

![Chart](https://quickchart.io/chart?c={%22type%22:%22bar%22,%22data%22:{%22labels%22:[%22openai/gpt-3.5-turbo%22,%22ai21/j2-jumbo-instruct%22,%22cohere/command-nightly%22,%22Wizard-Vicuna-13B-Uncensored%22,%22vicuna-7B-1.1%22,%22Manticore-13B%22,%22Guanaco-13B%22,%22WizardLM-13B-1.0%22,%22WizardLM-7B-Uncensored%22,%22Starchat%22],%22datasets%22:[{%22label%22:%22Python%22,%22data%22:[65,55,48,31,51,47,41,53,54,40]},{%22label%22:%22JavaScript%22,%22data%22:[8,62,39,45,44,37,37,52,37,45]}]}})

## Data Sets

Interview questions v1: https://huggingface.co/datasets/mike-ravkine/can-ai-code_junior-dev_v1

Interview results v1: SOON

## Repository Structure

* `junior-dev/*.yaml` - Interview questions (multi-language)
* `prompts/*.txt` - System prompts for the various models
* `prepare.py` - Specializes question into prompts for a specific language
* `interview-langchain.py` - Use a LangChain LLM model to write code
* `interview-starchat.py` - Use a Huggingface Space running Starchat model to write code
* `interview-gptq-modal.py` - Use GPTQ on Modal GPU rental platform
* `intreview-llamacpp.sh` - Use a GGML llama.cpp model to write code
* `evaluate.py` - Run tests for the generated code in a sandbox and grades each answer
* `report.py` - (WIP - not yet complete) Compare results from multiple interviews

## Question Format

A set of interview questions is a folder of .yaml files.  Each Question is a top-level key:

```yaml
SanityList:
    Request: "Write a {{language}} function things() that returns a list with three values: the number 5, the string 'foobar', the capital city of Spain."
    Description: "List function, see if the model can combine input facts with internal knowledge."
    Checks:
        input_name:
            assert: "f.name"
            eq: "things"
```

In this example `SanityList` is the name of the interview question.

`Request` will be turned into a prompt by replacing {{language}} with "javascript" or "python"
`Description` is a human-readable explanation of why this test is useful
`Checks` defines the expected behavior of the output.

### Checks and the 'f' object

Each check has a name, some `assert` value (python code) and an expected `eq` value.

The f object represents the sandbox view of the function.  Static analysis is performed on the function signature to extract the `f.name` and `f.args` fields, while `f.call` allows for function evaluation.

## Using this Repository

1. Prepare prompts for a python interview:

```bash
./prepare.py --language python --questions python.csv
```

2. Execute the interview.

With ChatGPT (gpt-3.5-turbo):

```bash
export OPENAI_API_KEY=...
./interview-langchain.py --model openai/chatgpt --questions python.csv --outdir results/chatgpt/
```

With Vicuna 1.1 (llama.cpp):

First open `interview-llamacpp.sh` and customize with your hostname and binary paths.  Then:

```bash
export PROMPT=prompts/Vicuna-1p1.txt
export MODEL=".../models/v3/ggml-vicuna-7b-1.1-q5_0.bin"
export OUTPUT="results/vicuna-1.1-7b/"
export INTERVIEW="python.csv"

./interview-llamacpp.sh
```

With VicUnlocked-30B-LoRA-GGML (GPTQ on Modal):

```bash
modal run -q ./interview-gptq-modal.py --outdir results/vicunlocked-30b/ --params model_parameters/precise.json --template prompts/Vicuna-1p1.txt --questions intervi
python.csv
```

3. Evaulate the results

```bash
./evaluate.py --language python --answers results/chatgpt/
```

## Interview format

The output of `prepare.py` is a simple csv with three columns: name, language and prompt

To create your own interview, simply feed the prompts to your model of choice and saveeach model outputs as name.txt in a results directory.  That's it!  Now you can perform evaluation.

# Roadmap / Future Work

Contributions are welcome!  Especially looking for additional interview sets and improvements to questions - open a PR! 

* Evaluate 30B and 65B open langauge models
* If the models are offered error messages or failing test results, could they produce better code?
* Can tweaking prompts improve performance?
