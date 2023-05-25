#!/bin/bash

#CRD716/ggml-vicuna-1.1-quantized
#https://huggingface.co/TheBloke/MPT-7B-Instruct-GGML
#https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML

#https://huggingface.co/TheBloke/wizardLM-7B-GGML
#https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML

INTERVIEW=python.csv

#PROMPT=prompts/Wizard-Vicuna.txt
#MODEL="/home/miner/ai/models/v3/Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_0.bin"
#OUTPUT="results/wizard-vicuna/"

PROMPT=prompts/Vicuna-1p1.txt
MODEL="/home/miner/ai/models/v3/ggml-vicuna-7b-1.1-q5_0.bin"
OUTPUT="results/vicuna-1.1-7b/"

SSH=miner
MAIN="~/ai/latest/main"
ARGS="--threads 4 --ctx_size 2048"
PARAMS="--n_predict 512 --temp 0.7 --top_k 40 --top_p 0.1 --repeat_last_n 256 --batch_size 1024 --repeat_penalty 1.17647"


mkdir -p $OUTPUT

# Read the CSV file line by line
while IFS= read -r line; do
    # Check if it's the first line (header row)
    if [[ $header_skipped != 1 ]]; then
        header_skipped=1
        continue
    fi

    # Split the line by comma
    IFS="," read -r col1 col2 <<< "$line"

    # Remove leading/trailing quotes from the second column
    col2=${col2%\"}
    col2=${col2#\"}

    # Expand the prompt template into a temporary file
    temp_file=$(mktemp)
    cat $PROMPT | sed "s/{{prompt}}/$col2/" > $temp_file

    CMDLINE="$MAIN $ARGS $PARAMS --model $MODEL --file $temp_file" 

    echo "Starting $col1"

    scp $temp_file $SSH:$temp_file
    ssh $SSH "$CMDLINE" < /dev/null | tee "${OUTPUT}${col1}.txt"

    echo "Done $col1"
    rm $temp_file
done < "$INTERVIEW"