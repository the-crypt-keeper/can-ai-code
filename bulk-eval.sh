#!/bin/bash

if [ "$1" == "" ]; then
    GLOB="results/interview_junior*.ndjson"
else
    GLOB="$1/interview_junior*.ndjson"
fi

# Search for files matching "results/interview_junior-dev*.ndjson"
for file in $GLOB; do
    echo $file

    # Extract the corresponding eval file name
    eval_file="${file/interview/eval}"

    # Check if the corresponding eval file exists
    if [ ! -f "$eval_file" ]; then
        # Execute ./evaluation.py with the input filename
        ./evaluate.py --input "$file"
    else
        echo "Already evaluated $file"
    fi
done
