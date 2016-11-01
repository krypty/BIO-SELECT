#!/bin/bash

if [ -z "$1" ]
then
	echo "No notebook provided. Exiting.."
	exit 1
fi

start_date=`date`
echo "Activate virtualenv..."
source `which virtualenvwrapper.sh`
workon TM_py2

notebook="$1"
notebook_html=${notebook/ipynb/html}
logfile="jupyter.log"
recipiant="gmarigliano93@gmail.com"

echo "Executing notebook and send it by mail..."
if jupyter nbconvert --ExecutePreprocessor.timeout=None --to=html --execute $notebook  1> $logfile 2> $logfile ; then
	cat $logfile | mutt -s "Jupyter success [$start_date]" -a $notebook_html -- $recipiant
else
	cat $logfile | mutt -s "Jupyter fail [$start_date]" -- $recipiant
fi

