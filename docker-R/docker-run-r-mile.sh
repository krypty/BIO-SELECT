#!/bin/bash
docker run -it -v `pwd`/dataset:/dataset --rm rdocker Rscript --no-save --no-restore --verbose limma-mile.R
