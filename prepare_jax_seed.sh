#!/bin/bash
# set -x
set -e

source "./utils.sh"

######################## Seed project specific ########################
ORG_REPO="jax-ml/jax" # Update to your github org/repo
generate_lock_file_path() {
  local python_version="$1"
  local final_path="build/requirements_lock_${PYTHON_VERSION//./_}.txt" # Update the path format regex
  echo "$final_path"
}

####################### General seed preparation #######################
# This script takes two inputs as command-line arguments.
# Check if exactly two arguments were provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 jax_version_tag_or_commit python_version"
  echo "Example: $0 jax-v0.6.2 3.10"
  exit 1
fi
# Assign the command-line arguments to variables
TAG_NAME="$1" # e.g., jax-v0.6.2
PYTHON_VERSION="$2" # e.g., 3.10

# Set $COMMIT_HASH if a commit is found
get_commit_hash_for_tag $TAG_NAME $ORG_REPO
echo "$COMMIT_HASH"

# Finalize the requirements lock file remote github path
requirements_lock_file_path=$(generate_lock_file_path "$PYTHON_VERSION")
final_seed_file_path="https://raw.githubusercontent.com/${ORG_REPO}/${COMMIT_HASH}/${requirements_lock_file_path}"
echo "final_seed_file_path=$final_seed_file_path"
# Download the python seed lock file
download_remote_file "$final_seed_file_path"

# for python_version in $PYTHON_VERSIONS; do
#   generate_pyproject_toml $python_version
# done
# Initialize pyproject.toml file with python version
# generate_pyproject_toml "$PYTHON_VERSION"

