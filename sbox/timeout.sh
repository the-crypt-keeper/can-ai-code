#!/bin/bash

echo -e "$WRAPPER_SOURCE" > /wrapper
echo -e "$ANSWER_SOURCE" > /answer

timeout --foreground 5 $@
CODE="$?"
if [ "$CODE" -eq "124" ]; then
   echo "### { \"error\": \"timeout!\" }"
fi