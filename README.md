# BIOSELECT

*Note: all the datasets are not provided in this repository since they are very heavy*

## How to get it ?

### Using Docker (recommanded)

**On GNU/Linux**
``` bash
./build-docker.sh
```

Then you can run Jupyter to work on the notebooks
``` bash
./run-jupyter-docker.sh
```

Or run a single one or all notebooks and get the results in **HTML in the output folder**
``` bash
./run-notebook-docker.sh <notebook_to_execute> <dataset>
# Example : ./run-notebook-docker.sh pipeline.ipynb Golub
# Datasets available : Golub, MILE or EGEOD22619. All datasets can be found in datasets/DatasetLoader.py

# or you can run all of them (only if you have enough coffee...)
./run-all-notebooks-docker.sh <dataset>
```

**On Windows**

You must use [Docker for Windows](https://docs.docker.com/docker-for-windows/)
```
docker build -t bioselect .
docker run -it -v $pwd\:/code -p 8888:8888 --rm bioselect
```

### Using a virtual environment
The following instructions are made for Ubuntu-like systems.

_Note: I assume you already have downloaded this project and the datasets_

1. `sudo apt-get update && apt-get install python-pip python-dev git`
2. `pip install --user virtualenvwrapper`
3. In your `~/.bashrc` add the following lines:
``` bash
export WORKON_HOME=~/.virtualenvs
mkdir -p $WORKON_HOME
source ~/.local/bin/virtualenvwrapper.sh
```
4. `mkvirtualenv --python=/usr/bin/python2 TM_py2`
5. `workon TM_py2`
6. `cd path/to/the/project`
7. `pip install -r requirements.txt`
8. `jupyter-notebook`
