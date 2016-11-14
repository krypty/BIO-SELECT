#!/bin/bash

docker run -it -v `pwd`:/code -p 8888:8888 --rm  bioselect
