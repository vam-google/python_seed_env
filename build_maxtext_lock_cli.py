#!/usr/bin/env python3

import argparse
import generate_seed_env_lock_files
import os
import prepare_jax_seed
import shutil
import sys
import utils

# --- Configuration Constants (matching shell script's globals) ---
GITHUB_ORG = "AI-Hypercomputer"
GITHUB_REPO = "maxtext"
# Name of the requirements file in the GitHub
REQUIREMENTS_FILE_NAME = "requirements.txt"
# Default Python versions to process if not specified via CLI
PYTHON_VERSIONS_DEFAULT = ["3.10", "3.11", "3.12"]
LATEST_JAX_VERSION = "jax-v0.6.2"
# Patch file path for JAX on Python 3.10 version
JAX_PATCH_FILE = "jax_requirements_lock_3_10.patch"
# Name of the GPU constrains file
CONSTRAINTS_GPU_ONLY = "constraints_gpu_only.txt"
# Name of the TPU constrains file
CONSTRAINTS_TPU_ONLY = "constraints_tpu_only.txt"


# --- Main CLI Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="A CLI tool to generate the MaxText requirements lock files for different "
                    "Python versions using uv."
    )

    # Add the optional maxtext_github_commit argument
    parser.add_argument(
        "--maxtext-github-commit",
        nargs='?',  # Optional, allows 0 or 1 argument
        type=str,
        default="main", # Default to 'main' branch if no commit is provided
        help="Optional: MaxText GitHub commit hash or branch name (e.g., 'main', "
             "'d3f08713bc8cc2700851c61c55d7d7dde1de5a02'). Defaults to 'main'."
    )

    # Add the optional jax_github_commit argument
    parser.add_argument(
        "--jax-github-commit-or-version",
        nargs='?',  # Optional, allows 0 or 1 argument
        type=str,
        default=LATEST_JAX_VERSION, # Default to LATEST_JAX_VERSION branch if no commit is provided
        help="Optional: JAX GitHub commit hash or versino (e.g., 'jax-v0.6.2' (version), "
             "'d3f08713bc8cc2700851c61c55d7d7dde1de5a02' (commit)). Defaults to LATEST_JAX_VERSION."
    )

    # Add the optional python-versions argument
    parser.add_argument(
        "--python-versions",
        nargs='*', # Optional, allows 0 or more arguments (space-separated list)
        default=PYTHON_VERSIONS_DEFAULT,
        help=f"Optional: Space-separated list of Python versions to generate lock files for. "
             f"Defaults to: {' '.join(PYTHON_VERSIONS_DEFAULT)}"
    )

    args = parser.parse_args()

    # Determine the remote URL for requirements.txt based on the commit/branch
    if args.maxtext_github_commit != "main":
        if not utils.is_valid_commit(args.maxtext_github_commit, "{GITHUB_ORG}/{GITHUB_REPO}"):
            print(f"Error: Provided commit/branch '{args.maxtext_github_commit}' is not valid. Exiting.", file=sys.stderr)
            return 1

    maxtext_remote_url = (
        f"https://raw.githubusercontent.com/{GITHUB_ORG}/{GITHUB_REPO}/"
        f"{args.maxtext_github_commit}/{REQUIREMENTS_FILE_NAME}"
    )

    # 1. Download the MaxText requirements.txt file
    try:
        utils.download_remote_file(maxtext_remote_url)
    except Exception as e:
        print(f"Fatal error during initial requirements file processing: {e}", file=sys.stderr)
        return 1

    # 2. Loop through Python versions to build lock files
    for python_version in args.python_versions:
        # Sanitize version string for file names (e.g., "3.10" -> "3_10")
        py_version_sanitized = python_version.replace('.', '_')
        for machine_type in ('tpu', 'gpu'):
            output_maxtext_requirement_lock_file = f"maxtext_requirements_lock_{machine_type}_{py_version_sanitized}.txt"
            jax_temp_lock_file = f"requirements_lock_{py_version_sanitized}.txt"

            print(f"\nProcessing for Python {python_version} on {machine_type.upper()}...")

            # Cleanup existing temporary files for this iteration to ensure a clean slate
            files_to_clean_per_iteration = [
                "uv.lock", # uv creates this
                output_maxtext_requirement_lock_file,
                "pyproject.toml",
                jax_temp_lock_file,
            ]
            for f_name in files_to_clean_per_iteration:
                if os.path.exists(f_name):
                    os.remove(f_name)
                    print(f"Cleaned up: '{f_name}'")

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
                # maxtext/seed_env_files/py${python_version}/
                output_folder = os.path.join("maxtext", "seed_env_files", f"py{py_version_sanitized}")
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

if __name__ == "__main__":
    # the exit status (0 for success, non-zero for error).
    sys.exit(main())

def apply_patch():
    """
    Apply patch to JAX requirements_lock.txt for Python 3.10.
    """
    pass
