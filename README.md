# SNIIRAM-exploration

Library that offers util abstractions to explore data extracted
using SNIIRAM-featuring.

Clone this repo and add it to the path to use it in notebooks.

## Requirements

This needs python 3.5.3 or above.

Make sure that you have a requierments-dev based active environnement.

    conda create -n exploration python=3.5.3
    pip install -r requirements-dev.txt

## Running tests
On your dev environnement, just launch the following command in the root of the project:

    nosetests
    
## Development
The development cycle is opinionated. Each time you commit, git will
launch four checks before it allows you to finish your commit: 
1. Black: we encourage you to install it and integrate to your dev
tool such as Pycharm. Check this [link](https://github.com/ambv/black). We massively encourage
to use it with Pycharm as it will automatically 
2. Flake8: enforces some extra checks.
3. Testing using Nosetests.
4. Coverage checks if the minimum coverage is ensured.

After cloning, you have to run in the root of the repo:
    
    source activate exploration
    pre-commit install