#!/bin/bash
set -x
set -e

source "./utils.sh"

GITHUB_ORG="AI-Hypercomputer"
GITHUB_REPO="maxtext"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_VERSIONS="3.10 3.11 3.12"
LATEST_JAX_VERSION="jax-v0.6.2"

# Inputs
# maxtext_github_commit (Optional). MaxText does not have release versions.
#   We get the head commit by default and allow users to specify a commit as an input.

############### Download and modify requirements.txt ###############
# Get the head commit at the github repo by default.
if [ "$#" -eq 0 ]; then
    maxtext_remote_url="https://raw.githubusercontent.com/${GITHUB_ORG}/${GITHUB_REPO}/main/${REQUIREMENTS_FILE}"
elif [ "$#" -eq 1 ]; then
    if is_valid_commit "$1" "${GITHUB_ORG}/${GITHUB_REPO}"; then
        maxtext_remote_url="https://raw.githubusercontent.com/${GITHUB_ORG}/${GITHUB_REPO}/${1}/${REQUIREMENTS_FILE}"
    else
        echo "Error"
        exit 1
    fi
else
    echo "Error"
    exit 1
fi

download_remote_file $maxtext_remote_url
# Fixing current maxtext requirements.txt
# TODO: Check if we should make the following less hard coded.
# TODO(kanglan): Update it in the maxtext repo and remove this block. Recommend to have lower bounds for each dep in the requirements.txt
# For whatever reason MaxText pins protobuf dependency to 3.20.3 which is not even supported anymore
# Note, it is better to let tensorflow (which is a dependency here) to define protobuf dep)
sed -i 's/protobuf==3.20.3/protobuf>=3.20.3/g' "$REQUIREMENTS_FILE"
sed -i 's/sentencepiece==0.1.97/sentencepiece>=0.1.97/g' "$REQUIREMENTS_FILE"
# All source links must be pinned to a hash, there is no other way to guarantee reproducibility
# for source links. They are a very small minority of deps, so imposing this requirement should be fine
sed -i 's/\/JetStream.git/\/JetStream.git@261f25007e4d12bb57cf8d5d61e291ba8f18430f/g' "$REQUIREMENTS_FILE"
sed -i 's/\/logging.git/\/logging.git@44b4810e65e8c0a7d9e4e207c60e51d9458a3fb8/g' "$REQUIREMENTS_FILE"
# This is a workaround for tensorflow-metadata version issue in python 3.10
echo "tensorflow-metadata>=1.14.0" >> requirements.txt
######################################################################

# TODO: Add an iteration for TPU/GPU
for python_version in $PYTHON_VERSIONS; do
    output_file="maxtext_requirements_lock_${python_version//./_}.txt"
    rm -f uv.lock
    rm -f "$output_file"
    rm -f pyproject.toml
    rm -f "requirements_lock_${python_version//./_}.txt"
    # Initialize a pyproject.toml file for the python version
    generate_pyproject_toml $python_version

    # Prepare jax seed file
    if [ $python_version == "3.10" ]; then
        ./prepare_jax_seed.sh $LATEST_JAX_VERSION $python_version
        # Add a patch if python version is 3.10
        patch requirements_lock_3_10.txt < jax_requirements_lock_3_10.patch
    else
        # This commit contains the required updated for jax cuda plugins.
        ./prepare_jax_seed.sh d3f08713bc8cc2700851c61c55d7d7dde1de5a02 $python_version
    fi

    # Run the uv commands to build project lock files
    # TODO: Refine this function
    build_seed_env "requirements_lock_${python_version//./_}.txt" "$REQUIREMENTS_FILE" "$output_file"

    # Save the generated files, i.e., pyproject.toml, maxtext_requirements_lock_<py version>.txt, and uv.lock to a folder
    # TODO: Assert those 3 files eixst
    # TODO: Add a subfoler for TPU/GPU
    output_folder="maxtext/seed_env_files/py${python_version/./}/"
    mkdir -p "$output_folder"
    mv uv.lock "$output_folder"
    mv pyproject.toml "$output_folder"
    mv "$output_file" "$output_folder"
done
