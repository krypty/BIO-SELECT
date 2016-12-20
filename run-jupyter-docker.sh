#!/bin/bash
docker run -it -e HOST_WD=`pwd` -v /data -v `pwd`:/code -v /var/run/docker.sock:/var/run/docker.sock -p 8888:8888 --rm bioselect
