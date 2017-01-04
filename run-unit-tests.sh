#!/usr/bin/env bash
rm -r tests/__pycache__ tests/__init__.pyc 2> /dev/null; docker run -it -v `pwd`:/code bioselect pytest -v .
