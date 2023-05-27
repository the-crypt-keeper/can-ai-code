# Can AI Code?

A self-evaluating interview for AI coding models.

## Key Ideas

* Interview questions written by humans, test taken by AI
* Sandbox enviroment (Docker-based) for untrusted Python and NodeJS execution
* Compare LLM models against each other
* For a given LLM, compare prompting techniques and hyper-parameters

## Repository Structure

* `junior-dev/*.yaml` - Interview questions (multi-language)
* `prepare.py` - Specializes question into prompts for a specific language
* `interview-langchain.py` - Use a LangChain LLM model to write code
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

1. Prepare prompts for an interview:

```bash
./prepare.py --language python --output python.csv
```

2. Execute the interview with ChatGPT (gpt-3.5-turbo)

```bash
export OPENAI_API_KEY=...
./interview-langchain.py --model openai/chatgpt --questions python.csv --outdir results/chatgpt/
```

3. Evaulate the results

```bash
./evaluate.py --language python --answers results/chatgpt/
```

## Interview format

The output of `prepare.py` is a simple csv with two columns: name and prompt

To create your own interview, simply feed the prompts to your model of choice and saveeach model outputs as name.txt in a results directory.  That's it!  Now you can perform evaluation.

## Future Work

* If the models are offered error messages or failing test results, could they produce better code?
* Can tweaking prompts improve performance?
