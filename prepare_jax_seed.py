#!/usr/bin/env python3

import os
import sys

# Import functions from the utils.py script
import utils

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

if __name__ == "__main__":
    print(f"Current working directory in script: {os.getcwd()}")

    # Check if exactly two arguments were provided
    if len(sys.argv) != 3:
        print("Usage: python3 prepare_jax_seed.py <jax_version_tag_or_commit> <python_version>", file=sys.stderr)
        print("Example: python3 prepare_jax_seed.py jax-v0.4.26 3.10", file=sys.stderr)
        sys.exit(1)

    # Assign the command-line arguments to variables
    TAG_NAME = sys.argv[1]
    PYTHON_VERSION = sys.argv[2]

    # Finalize the requirements lock file remote github path
    requirements_lock_file_name = generate_lock_file_path(PYTHON_VERSION)

    # Get the commit hash for the tag or directly validate if it's a commit
    # This now calls the function from the `utils` module
    COMMIT_HASH = utils.get_commit_hash_for_tag(TAG_NAME, ORG_REPO)
    # The echo of $COMMIT_HASH is done inside get_commit_hash_for_tag
    final_seed_file_path = f"https://raw.githubusercontent.com/{ORG_REPO}/{COMMIT_HASH}/{requirements_lock_file_name}"
    print(f"final_seed_file_path={final_seed_file_path}")
    # Download the python seed lock file
    # This now calls the function from the `utils` module
    utils.download_remote_file(final_seed_file_path)

    print("\nPrepared JAX seed successfully.")
