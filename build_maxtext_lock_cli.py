#!/usr/bin/env python3

import argparse
import os
import prepare_jax_seed
import shutil
import sys
import utils

# Configuration Constants
GITHUB_ORG = "AI-Hypercomputer"
GITHUB_REPO = "maxtext"
# Name of the requirements file in the GitHub
REQUIREMENTS_FILE_NAME = "requirements.txt"
# Supported python versions of the CLI tool
SUPPORTED_PYTHON_VERSIONS = {"3.11", "3.12"}
# Latest JAX version
LATEST_JAX_VERSION = "jax-v0.6.2"
# Patch file path for JAX on Python 3.10 version
JAX_PATCH_FILE = "jax_requirements_lock_3_10.patch"
# Name of the GPU constrains file
CONSTRAINTS_GPU_ONLY = "constraints_gpu_only.txt"
# Name of the TPU constrains file
CONSTRAINTS_TPU_ONLY = "constraints_tpu_only.txt"


def main():
    parser = argparse.ArgumentParser(
        description="A CLI tool to generate the MaxText requirements lock files for different "
                    "Python versions using uv."
    )

    # Add the required maxtext-github-commit argument
    parser.add_argument(
        "--maxtext-github-commit",
        type=str,
        required=True,
        help="MaxText GitHub commit hash or branch name (e.g., 'main' (branch), "
            "'d3f08713bc8cc2700851c61c55d7d7dde1de5a02' (commit))."
    )

    # Add the required github-commit-or-version argument
    parser.add_argument(
        "--jax-github-commit-or-version",
        type=str,
        required=True,
        help="JAX GitHub commit hash or version (e.g., 'jax-v0.6.2' (version), "
            "'d3f08713bc8cc2700851c61c55d7d7dde1de5a02' (commit))."
    )

    # Add the required python-versions argument (still allows multiple values)
    parser.add_argument(
        "--python-versions",
        nargs='+', # Required, allows 0 or more arguments (space-separated list)
        required=True,
        help=f"Space-separated list of Python versions to generate lock files for (e.g., '3.9 3.10')."
    )

    args = parser.parse_args()

    # Santity check for the input Python versions
    for python_version in args.python_versions:
        if not python_version in SUPPORTED_PYTHON_VERSIONS:
            print("Error: Provided unsupported python versions in --python-versions argument. " \
                  f"'{args.python_versions}' is not supported. Exiting.", file=sys.stderr)
            return 1

    # Determine the remote URL for requirements.txt based on the commit/branch
    if args.maxtext_github_commit != "main":
        if not utils.is_valid_commit(args.maxtext_github_commit, f"{GITHUB_ORG}/{GITHUB_REPO}"):
            print(f"Error: Provided commit/branch '{args.maxtext_github_commit}' is not valid. Exiting.", file=sys.stderr)
            return 1

    maxtext_remote_url = (
        f"https://raw.githubusercontent.com/{GITHUB_ORG}/{GITHUB_REPO}/"
        f"{args.maxtext_github_commit}/{REQUIREMENTS_FILE_NAME}"
    )

    # 1. Download the MaxText requirements.txt file
    try:
        utils.download_remote_file(maxtext_remote_url)
        # TODO: Once the MaxText requirements.txt gets cleanup, we can remove the dependency adjustment here.
        utils.fix_maxtext_requirements(REQUIREMENTS_FILE_NAME)
    except Exception as e:
        print(f"Fatal error during initial requirements file processing: {e}", file=sys.stderr)
        return 1

    # 2. Loop through Python versions to build lock files
    for python_version in args.python_versions:
        py_version_sanitized = python_version.replace('.', '_')
        jax_temp_lock_file = f"requirements_lock_{py_version_sanitized}.txt"

        # Clean up the existing JAX lock file before fetching it from JAX
        _cleanup_files([jax_temp_lock_file])

        # Loop through machine types (i.e., TPU and GPU)
        for machine_type in ('tpu', 'gpu'):
            output_maxtext_requirement_lock_file = f"maxtext_requirements_lock_{machine_type}_{py_version_sanitized}.txt"
            print(f"\nProcessing for Python {python_version} on {machine_type.upper()}...")

            # Cleanup existing temporary files for this iteration to ensure a clean slate
            files_to_clean_per_iteration = [
                "uv.lock", # uv creates this
                output_maxtext_requirement_lock_file,
                "pyproject.toml",
            ]
            _cleanup_files(files_to_clean_per_iteration)

            try:
                # Initialize pyproject.toml for the current Python version
                utils.generate_pyproject_toml(python_version)

                # Prepare JAX seed file based on Python version
                if python_version == "3.10":
                    # Use latest JAX version for python 3.10
                    prepare_jax_seed.run(LATEST_JAX_VERSION, python_version)
                    # Apply the specific patch for Python 3.10 JAX requirements
                    apply_patch(jax_temp_lock_file, JAX_PATCH_FILE)
                else:
                    # For other Python versions, use a specific JAX commit or version
                    prepare_jax_seed.run(args.jax_github_commit_or_version, python_version)

                # Build the combined seed environment lock files using uv
                if machine_type == 'tpu':
                    utils.build_seed_env(
                        seed_file = jax_temp_lock_file,
                        project_requirements_file = REQUIREMENTS_FILE_NAME,
                        output_file = output_maxtext_requirement_lock_file,
                        constraints_file = CONSTRAINTS_TPU_ONLY,
                    )
                elif machine_type == 'gpu':
                    utils.build_seed_env(
                        seed_file = jax_temp_lock_file,
                        project_requirements_file = REQUIREMENTS_FILE_NAME,
                        output_file = output_maxtext_requirement_lock_file,
                        constraints_file = CONSTRAINTS_GPU_ONLY,
                    )
                else:
                    print("\nEncounter machine_type that is not supported. Skipping.")
                    continue

                # Define the output folder structure
                # maxtext/seed_env_files/py${python_version}/{machine_type}
                output_folder = os.path.join("maxtext", "seed_env_files", f"py{py_version_sanitized}", f"{machine_type}")
                os.makedirs(output_folder, exist_ok=True) # Create folder if it doesn't exist

                # Move the generated files to the dedicated output folder
                files_to_move = [
                    "uv.lock", # This is generated by uv pip compile
                    "pyproject.toml",
                    output_maxtext_requirement_lock_file, # The main output lock file for MaxText
                ]

                for f_to_move in files_to_move:
                    if os.path.exists(f_to_move):
                        shutil.move(f_to_move, output_folder)
                        print(f"Moved '{f_to_move}' to '{output_folder}'")
                    else:
                        print(f"Warning: Expected file '{f_to_move}' not found for moving to '{output_folder}'.", file=sys.stderr)
            except Exception as e:
                print(f"Error processing Python {python_version}: {e}", file=sys.stderr)
                # TODO: Decide whether to continue or stop on error for a specific Python version
                # We are currently continue to the next Python version if encounter errors
                print(f"Skipping Python {python_version} due to error. Continuing with next version if any.", file=sys.stderr)
                continue

    print("\nCompleted building MaxText environment lock files.")

    return 0

def _cleanup_files(files_to_clean_per_iteration):
    for f_name in files_to_clean_per_iteration:
        if os.path.exists(f_name):
            os.remove(f_name)
            print(f"Cleaned up: '{f_name}'")

if __name__ == "__main__":
    # the exit status (0 for success, non-zero for error).
    sys.exit(main())

def apply_patch():
    """
    Apply patch to JAX requirements_lock.txt for Python 3.10.
    """
    pass
