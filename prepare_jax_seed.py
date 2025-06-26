#!/usr/bin/env python3

import os
import sys
from utils import download_seed_file_from_git_remote

# Seed project specific configurations
ORG_REPO = "jax-ml/jax" # Update to your github org/repo

def generate_lock_file_path(python_version):
    """
    Generates the path format for the requirements lock file.
    Equivalent to the shell's generate_lock_file_path function.

    Args:
        python_version (str): The Python version (e.g., "3.10").

    Returns:
        str: The formatted lock file path.
    """
    formatted_python_version = python_version.replace('.', '_')
    final_path = f"build/requirements_lock_{formatted_python_version}.txt"
    return final_path

def run(tag_or_commit: str, python_version: str):
    """
    Prepares the JAX seed environment by downloading the appropriate lock file.

    Args:
        tag_or_commit (str): The JAX version tag or commit hash (e.g., "jax-v0.6.2").
        python_version (str): The Python version (e.g., "3.10").
    """
    # Finalize the requirements lock file remote github path
    requirements_lock_file_name = generate_lock_file_path(python_version)

    # Download the python seed lock file
    download_seed_file_from_git_remote(ORG_REPO, tag_or_commit, requirements_lock_file_name)

    print("\nPrepared JAX seed successfully.")

if __name__ == "__main__":
    print(f"Current working directory in script: {os.getcwd()}")

    # Check if exactly two arguments were provided
    if len(sys.argv) != 3:
        print("Usage: python3 prepare_jax_seed.py <jax_version_tag_or_commit> <python_version>", file=sys.stderr)
        print("Example: python3 prepare_jax_seed.py jax-v0.6.2 3.10", file=sys.stderr)
        sys.exit(1)

    # Assign the command-line arguments to variables
    TAG_OR_COMMIT = sys.argv[1]
    PYTHON_VERSION = sys.argv[2]

    # Prepare teh JAX seed environment
    run(TAG_OR_COMMIT, PYTHON_VERSION)

    print("\nPrepared JAX seed successfully.")
