import argparse
import os
import shutil
import sys

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


# --- Main CLI Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="A CLI tool to generate the MaxText requirements lock files for different "
                    "Python versions using uv."
    )

    # Add the positional maxtext_github_commit argument
    parser.add_argument(
        "maxtext_github_commit",
        nargs='?',  # Optional, allows 0 or 1 argument
        type=str,
        default="main", # Default to 'main' branch if no commit is provided
        help="Optional: MaxText GitHub commit hash or branch name (e.g., 'main', "
             "'d3f08713bc8cc2700851c61c55d7d7dde1de5a02'). Defaults to 'main'."
    )

    # Add the positional jax_github_commit argument
    parser.add_argument(
        "jax_github_commit",
        nargs='?',  # Optional, allows 0 or 1 argument
        type=str,
        default="main", # Default to 'main' branch if no commit is provided
        help="Optional: JAX GitHub commit hash or branch name (e.g., 'main', "
             "'d3f08713bc8cc2700851c61c55d7d7dde1de5a02'). Defaults to 'main'."
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
        if not is_valid_commit(args.maxtext_github_commit, GITHUB_ORG, GITHUB_REPO):
            print(f"Error: Provided commit/branch '{args.maxtext_github_commit}' is not valid. Exiting.", file=sys.stderr)
            return 1

    maxtext_remote_url = (
        f"https://raw.githubusercontent.com/{GITHUB_ORG}/{GITHUB_REPO}/"
        f"{args.maxtext_github_commit}/{REQUIREMENTS_FILE_NAME}"
    )

    # 1. Download the MaxText requirements.txt file
    try:
        download_remote_file(maxtext_remote_url, REQUIREMENTS_FILE_NAME)
    except Exception as e:
        print(f"Fatal error during initial requirements file processing: {e}", file=sys.stderr)
        return 1

    # 2. Loop through Python versions to build lock files
    for python_version in args.python_versions:
        # Sanitize version string for file names (e.g., "3.10" -> "3_10")
        py_version_sanitized = python_version.replace('.', '_')
        output_maxtext_requirement_lock_file = f"maxtext_requirements_lock_{py_version_sanitized}.txt"
        jax_temp_lock_file = f"requirements_lock_{py_version_sanitized}.txt"

        print(f"\nProcessing for Python {python_version}...")

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
            generate_pyproject_toml(python_version)

            # Prepare JAX seed file based on Python version
            if python_version == "3.10":
                # Use latest JAX version for python 3.10
                prepare_jax_seed(LATEST_JAX_VERSION, python_version, jax_temp_lock_file)
                # Apply the specific patch for Python 3.10 JAX requirements
                apply_patch(jax_temp_lock_file, JAX_PATCH_FILE)
            else:
                # For other Python versions, use a specific JAX commit or version
                prepare_jax_seed(args.jax_github_commit, python_version, jax_temp_lock_file)

            # Build the combined seed environment lock files using uv
            build_seed_env(jax_temp_lock_file, REQUIREMENTS_FILE_NAME, output_maxtext_requirement_lock_file)

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
            # Decide whether to continue or stop on error for a specific Python version
            print(f"Skipping Python {python_version} due to error. Continuing with next version if any.", file=sys.stderr)
            continue # Continue to the next Python version

    print("\n--- CLI process completed. ---")

    return 0

if __name__ == "__main__":
    # the exit status (0 for success, non-zero for error).
    sys.exit(main())

# Assuming these functions will be provided in the utility package.
# --- Utility Functions (approximating utils.sh and other script behaviors) ---
def is_valid_commit(commit_hash: str, repo_owner: str, repo_name: str) -> bool:
    """
    Checks if a commit hash or branch name is valid for a given repository.
    """
    pass

def download_remote_file(url: str, local_path: str):
    """Downloads a file from a URL to a specified local path."""
    pass

def generate_pyproject_toml(python_version: str, file_path: str = "pyproject.toml"):
    """
    Generates a basic pyproject.toml file for the specified Python version.
    """
    pass

# Assuming these functions will be provided in the prepare jax seed package.
# --- JAX Seed Utility Functions (approximating prepare_jax_seed.sh and other script behaviors) ---
def prepare_jax_seed(jax_version_or_commit: str, python_version: str,
                     output_jax_lock_file: str):
    """
    Simulates the behavior of the `prepare_jax_seed.sh` script.
    This function creates a JAX requirements lock file.
    """
    pass

# Assuming these functions will be provided in the build seed package.
# --- Build Seed/Host Env Utility Functions (approximating build_seed_env.sh and other script behaviors) ---
def build_seed_env(jax_lock_file: str, main_requirements_file: str, final_output_file: str):
    """
    Build Host/Seed Environment for MaxText.
    Generates the maxtext_requirements_lock_3_12.txt
    """
    pass

def apply_patch():
    """
    Apply patch to JAX requirements_lock.txt for Python 3.10.
    """