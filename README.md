# BIOSELECT

## How to get it ?

### Using Docker (recommanded)

**On GNU/Linux**
```
docker build -t bioselect .
docker run -it -v `pwd`:/code -p 8888:8888 --rm bioselect
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
