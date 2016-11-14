#!/bin/bash

if [ -z "$1" ]
then
	echo "Invalid arguments"
	echo "Usage: $0 <dataset>"
	echo "Please read README.md to see examples"
	echo "Exiting..."
	exit 1
fi

dataset=$1

files=(*.ipynb)

for notebook in ${files[*]}
do
	echo "Running $notebook with dataset $dataset"
	./run-notebook-docker.sh $notebook $dataset  
done
