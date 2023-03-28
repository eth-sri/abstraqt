#!/bin/bash
# Installs abstraqt using conda

# enable bash strict mode
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

#################
# PREPARE CONDA #
#################

set +u
# enable using conda from within this script
eval "$(conda shell.bash hook)"
set -u

######################
# CREATE ENVIRONMENT #
######################

CONDA_ENV="abstraqt"

echo "Removing potentially outdated old environment..."
conda deactivate
conda env remove --name $CONDA_ENV

echo "Creating and activating new environment..."
conda create --yes --name $CONDA_ENV python=3.8
conda activate $CONDA_ENV

#############################
# SET ENVIRONMENT VARIABLES #
#############################

echo -e "\nSetting environment variables to enable default logging..."
conda env config vars set DEFAULT_LOGGING=DEBUG

conda activate $CONDA_ENV

###########
# INSTALL #
###########

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/install-pip.sh" "$@"
