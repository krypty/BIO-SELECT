# BIOSELECT

*Note: all the datasets are not provided in this repository since they are very heavy*.

## Master thesis presentation (Google Slides)

You can find the presentation I have done (in French) [here](https://docs.google.com/presentation/d/1Cf2wGaCPGiQ0eqaUWuwIpMAMycqNcf5OZA9eYOQiFnc/edit?usp=sharing)

## Master thesis report

You can find the report I have done (in French) in the [report](/report/report.pdf) folder.

## Setup the project using Docker (recommended)

### Requirements
* You will need a GNU/Linux distribution with Docker support. Xubuntu 16.04 is recommended.
* You need to have Docker installed


``` bash
./build-docker.sh
```

Then you can run Jupyter to work on the notebooks
``` bash
./run-jupyter-docker.sh
```

To run the unit tests
``` bash
./run-unit-tests.sh
```

### Setup the project using Python virtualenv

**Warning**: You will not be able to run Limma if you use this method.

**Information**: You don't need to follow these instructions if you followed the Docker ones

### Requirements
* A GNU/Linux operating system, it is assumed that you use an Ubuntu-like distribution
* Git
* Python 2 and pip

#### Get the project
``` bash
git clone https://github.com/krypty/BIO-SELECT.git
cd BIO-SELECT

# If you don't have pip installed
wget https://bootstrap.pypa.io/get-pip.py
sudo python2 get-pip.py
```

#### Create and activate the virtual environment
``` bash
sudo pip install virtualenv
virtualenv -p python2 bioselect
source bioselect/bin/activate
```

#### Install the dependancies
``` bash
# Those packages need to be installed on the system because some librairies require them
sudo apt-get update && sudo apt-get install build-essential python-dev

# Install python dependencies
pip install -r requirements.txt
```

#### Run a Jupyter notebook
To see if everything has been installed correctly, start Jupyter and run a notebook:
``` bash
jupyter-notebook
```

Now open your web browser at `http://localhost:8888` and
open a Juypter notebook like `DatasetVisualisation.ipynb` for instance.

## Getting started
The project is composed of Jupyter notebooks and Python classes.
The notebooks shows the graphs and the general workflow of the project and use the Python classes
in the background.

The first notebook you might want to check out is the one called `features_selection.ipynb`.
It contains the dataset loadings and executes the algorithms. To see what happens behind the scenes
you should look at the `import` statements and what is going on in these classes.

An other thing you can do to getting started is to review the unit tests. It can help you to understand how a class work
