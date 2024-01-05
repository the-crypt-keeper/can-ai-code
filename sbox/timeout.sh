#!/bin/bash

echo -e "$WRAPPER_SOURCE" | base64 -d > /wrapper
echo -e "$ANSWER_SOURCE" | base64 -d > /answer

timeout --foreground 5 $@
CODE="$?"
if [ "$CODE" -eq "124" ]; then
   echo "### { \"error\": \"timeout!\" }"
fi
