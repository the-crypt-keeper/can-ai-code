#!/bin/bash

# Search for files matching "results/interview_junior-dev*.ndjson"
for file in results/interview_junior*.ndjson; do
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
