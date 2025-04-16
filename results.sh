#!/bin/bash

if [ "$1" == "" ]; then
  WILDCARD='results/eval*'
else
  WILDCARD=$1
fi

STREAMLIT_SERVER_ADDRESS=localhost streamlit run app.py "$WILDCARD"
