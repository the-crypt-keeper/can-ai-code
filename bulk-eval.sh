#!/bin/bash

if [ "$1" == "" ]; then
    GLOB="results/interview_*.ndjson"
else
    GLOB="$1/interview_*.ndjson"
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
        INTERVIEW=`basename $file | cut -d'_' -f 2`
        ./evaluate.py --interview ${INTERVIEW} --input "$file"
    else
        echo "Already evaluated $file"
    fi
done
