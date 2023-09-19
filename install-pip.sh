#!/bin/bash
# Installs abstraqt using pip

# enable bash strict mode
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

###################
# PARSE ARGUMENTS #
###################

FAST=${1:-}

################
# INSTALLATION #
################

echo -e "\nInstalling abstraqt..."
pip --version
pip install git+https://github.com/Quantomatic/pyzx@0f8737d

# --upgrade: upgrade all packages to the newest available version
# --upgrade-strategy=eager: dependencies are upgraded regardless of whether the
# currently installed version satisfies the requirements
# --editable: Install in editable mode from the local project path
# [test]: also install test dependencies
time pip install --upgrade --upgrade-strategy=eager --editable .[test]

###########
# CLEANUP #
###########

if [ "$FAST" != "--fast" ]; then

	echo -e "\nCleaning up potentially outdated cache..."
	python -m abstraqt.utils.cache_dir --clean

fi

#########
# TESTS #
#########

if [ "$FAST" != "--fast" ]; then

	echo -e "\n\nRunning tests to check if installation was successful..."
	pytest --version
	time pytest --import-mode=importlib

fi
