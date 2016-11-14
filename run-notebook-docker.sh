#!/bin/bash

if [ -z "$2" ]
then
	echo "Invalid arguments"
	echo "Usage: $0 <notebook_to_execute> <dataset>"
	echo "Please read README.md to see examples"
	echo "Exiting..."
	exit 1
fi

notebook="$1"
notebook_prefix=${notebook/.ipynb/}

dataset="$2"

docker run -it -e "BIOSELECT_DATASET=${dataset}" -v `pwd`:/code -p 8889:8888 --rm  bioselect jupyter nbconvert --ExecutePreprocessor.timeout=None --to=html --execute --output="${notebook_prefix}_${dataset}_`date +%F-%T`" --output-dir=outputs $notebook
