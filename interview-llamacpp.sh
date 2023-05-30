#!/bin/bash

MODEL="${MODEL:-/home/miner/ai/models/v3/Manticore-13B.ggmlv3.q5_0.bin}"
INTERVIEW="${INTERVIEW:-python.csv}"
PROMPT="${PROMPT:-prompts/Manticore.txt}"
OUTPUT="${OUTPUT:-results/manticore-13b/}"

echo "INTERVIEW = $INTERVIEW"
echo "MODEL = $MODEL, PROMPT = $PROMPT"
echo "OUTPUT = $OUTPUT"

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
    IFS="," read -r col1 lang col2 <<< "$line"

    # Skip existing files
    if [[ -e "${OUTPUT}${col1}.txt" ]]; then
        echo "${col1} already done"
        continue
    fi

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