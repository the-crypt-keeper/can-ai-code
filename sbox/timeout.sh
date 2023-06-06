#!/bin/bash

timeout --foreground 5 $@
CODE="$?"
if [ "$CODE" -eq "124" ]; then
   echo "### { \"error\": \"timeout!\" }"
fi
