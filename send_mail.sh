#!/bin/bash

# stop on first error
#set -e

echo "Activate virtualenv..."
source `which virtualenvwrapper.sh`
workon TM_py2

notebook="DatasetsVisualization.ipynb"
notebook_html=${notebook/ipynb/html}
logfile="jupyter.log"
recipiant="gmarigliano93@gmail.com"

echo "Executing notebook and send it by mail..."
if jupyter nbconvert --ExecutePreprocessor.timeout=None --to=html --execute $notebook  1> $logfile 2> $logfile 2> $logfile ; then
	cat $logfile | mutt -s "Jupyter success" -a $notebook_html -- $recipiant
else
	cat $logfile | mutt -s "Jupyter fail" -- $recipiant
fi
#mutt -s "Jupyter" -- gmarigliano93@gmail.com && echo "Result in HTML" | mutt -s "Jupyter" -a "DatasetsVisualization.html" -- gmarigliano93@gmail.com

