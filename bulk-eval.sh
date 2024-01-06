#!/bin/bash

if [ "$INTERVIEW" == "" ]; then
    INTERVIEW="junior-v2"
fi

if [ "$1" == "" ]; then
    GLOB="results/interview_${INTERVIEW}*.ndjson"
else
    GLOB="$1/interview_${INTERVIEW}*.ndjson"
fi

# Search for files matching "results/interview_junior-dev*.ndjson"
for file in $GLOB; do
    echo $file

    # Extract the corresponding eval file name
    eval_file="${file/interview/eval}"
    if [ "$ALWAYS" == "1" ]; then
      eval_file=""
    fi

    # Check if the corresponding eval file exists
    if [ ! -f "$eval_file" ]; then
        # Execute ./evaluation.py with the input filename
        ./evaluate.py --interview ${INTERVIEW} --input "$file"
    else
        echo "Already evaluated $file"
    fi
done
