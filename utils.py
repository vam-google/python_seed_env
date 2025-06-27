import requests
import sys
import os
import re
import subprocess
import json
import fileinput


def download_remote_file(file_url):
    """
    Downloads a remote file from the given URL.

    Args:
        file_url (str): The URL of the file to download.

    Raises:
        SystemExit: If there's an error during the existence check or download.
    """
    try:
        # Checking existence of the remote file. Perform a HEAD request.
        response = requests.head(file_url, allow_redirects=True, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors

        if response.status_code == 404:
            print(f"Error: File not found (HTTP 404) at {file_url}.", file=sys.stderr)
            sys.exit(1)
        elif response.status_code != 200:
            print(f"Error: Unexpected HTTP status code {response.status_code} for {file_url}. Expected 200 OK.", file=sys.stderr)
            sys.exit(1)

        # Extract filename from URL
        file_name = os.path.basename(file_url)
        if not file_name:  # Handle cases where URL might not have a simple filename
            print(f"Error: Could not determine filename from URL: {file_url}", file=sys.stderr)
            sys.exit(1)

        # Download the file
        print(f"Downloading {file_url} to {file_name}...")
        with requests.get(file_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded {file_name}")

    except requests.exceptions.RequestException as e:
        print(f"Error: Network or HTTP error during download of {file_url}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def generate_pyproject_toml(python_version, output_file="./pyproject.toml"):
    """
    Generates a basic pyproject.toml file for a given Python version.

    Args:
        python_version (str): The target Python version (e.g., "3.10", "3.11").
        output_file (str, optional): The path to the output pyproject.toml file.
                                     Defaults to "./pyproject.toml".

    Returns:
        int: 0 on success, 1 on error.
    """
    if not python_version:
        print("Error: Python version is required.", file=sys.stderr)
        print("Usage: generate_pyproject_toml <python_version> [output_file]", file=sys.stderr)
        return 1

    # Basic validation for python_version format (e.g., "3.10", not just "3")
    # Using re.match to ensure the pattern matches from the beginning of the string
    if not re.match(r"^[0-9]+\.[0-9]+$", python_version):
        print(f"Warning: Python version '{python_version}' does not seem to be in 'X.Y' format.", file=sys.stderr)
        return 1

    content = f"""\
[project]
name = "test_env_1"
version = "0.1.0"
requires-python = "=={python_version}.*"
dependencies = [
]
"""

    try:
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"Successfully generated '{output_file}' for Python version '{python_version}'.")
        return 0
    except IOError as e:
        print(f"Error: Failed to generate '{output_file}'. Reason: {e}", file=sys.stderr)
        return 1

def _run_command(command_list, cwd=None, capture_output=False, text=False, check=True):
    """
    Helper function to run a shell command and handle errors.

    Args:
        command_list (list): The command to run as a list of arguments (e.g., ['uv', 'add', '-r', 'file.txt']).
        cwd (str, optional): The current working directory for the command.
        capture_output (bool): Whether to capture stdout and stderr.
        text (bool): Decode stdout/stderr as text.
        check (bool): If True, raise a CalledProcessError if the command returns a non-zero exit code.

    Returns:
        subprocess.CompletedProcess: The result of the command execution.
    """
    if not isinstance(command_list, list):
        raise TypeError(f"Command must be a list of arguments, but received type {type(command_list).__name__}: {command_list}")
    if not command_list:
        raise ValueError("Command list cannot be empty.")
    
    print(f"Executing: {' '.join(command_list)}", file=sys.stderr)
    return subprocess.run(command_list, cwd=cwd, capture_output=capture_output, text=text, check=check)

def build_seed_env(seed_file, project_requirements_file, output_file,
                   constraints_file=""):
    """
    Builds a Python environment using uv, managing dependencies based on seed
    and project requirements, and handling TPU constraints.

    Args:
        seed_file (str): Path to the initial seed requirements file.
        project_requirements_file (str): Path to the main project requirements file.
        output_file (str): Path for the final exported project lock file.
        constraints_file (str, optional): Path to the constraints file to exclude useless deps from seed.
                                          Defaults to "".

    Returns:
        int: 0 on success, 1 on failure.
    """
    print(f"Building seed environment with seed: {seed_file}; project requirements: {project_requirements_file}")
    print(f"Output file: {output_file}; constraints: {constraints_file}")

    try:
        # ---------------------------------
        print("Removing uv.lock if it exists...")
        if os.path.exists("uv.lock"):
            os.remove("uv.lock")

        print(f"Adding seed dependencies from {seed_file}...")
        _run_command([
            "uv", "add", "--managed-python", "--no-build", "--no-sync",
            "--resolution=highest", "-r", seed_file
        ])

        if constraints_file != "":
            # Handle constraints (equivalent to `cat ... | xargs uv remove ...`)
            print(f"Removing constraint dependencies from {constraints_file}...")
            if os.path.exists(constraints_file):
                with open(constraints_file, 'r') as f:
                    deps_to_remove = [line.strip() for line in f if line.strip()]
                
                if deps_to_remove:
                    print(f"Dependencies to remove from {constraints_file}: {', '.join(deps_to_remove)}")
                    for dep in deps_to_remove:
                        print(f"Attempting to remove: {dep}...")
                        _run_command([
                            "uv", "remove", "--managed-python", "--no-sync",
                            "--resolution=highest", dep
                        ])
                else:
                    print(f"No dependencies found in {constraints_file} to remove.")
            else:
                print(f"Warning: constraints file '{constraints_file}' not found. Skipping removal.", file=sys.stderr)

        print(f"Adding project requirements from {project_requirements_file}...")
        _run_command([
            "uv", "add", "--managed-python", "--no-sync",
            "--resolution=highest", "-r", project_requirements_file
        ])

        # uv export --managed-python --locked --no-hashes --no-annotate --resolution=highest --output-file="$output_file"
        print(f"Exporting highest resolution lock file to {output_file}...")
        _run_command([
            "uv", "export", "--managed-python", "--locked", "--no-hashes", "--no-annotate",
            "--resolution=highest", "--output-file", output_file
        ])

        # 6. python3 lock_to_lower_bound_project.py "$output_file" pyproject.toml
        print(f"Running lock_to_lower_bound_project.py with {output_file} and pyproject.toml...")
        _run_command([
            sys.executable, "lock_to_lower_bound_project.py", output_file, "pyproject.toml"
        ])

        print("Removing uv.lock before lowest resolution lock...")
        if os.path.exists("uv.lock"):
            os.remove("uv.lock")
        else:
            print(f"Warning: uv.lock does not exist, skipping removal.", file=sys.stderr)

        print("Running uv lock with lowest resolution...")
        _run_command([
            "uv", "lock", "--managed-python", "--resolution=lowest"
        ])

        print(f"Exporting lowest resolution lock file to {output_file}...")
        _run_command([
            "uv", "export", "--managed-python", "--locked", "--no-hashes", "--no-annotate",
            "--resolution=lowest", "--output-file", output_file
        ])

        print("Environment build process completed successfully.")
        return 0

    except TypeError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: Command not found or file missing: {e}. "
              "Ensure 'uv' is installed and in your PATH, "
              "and all constraint/requirement files exist.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Error: A command failed with exit code {e.returncode}.", file=sys.stderr)
        if e.stdout:
            print(f"STDOUT:\n{e.stdout.decode()}", file=sys.stderr)
        if e.stderr:
            print(f"STDERR:\n{e.stderr.decode()}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return 1

def _check_github_rest_api_message(response_json_string):
    """
    Checks a GitHub REST API JSON response for specific error messages,
    such as API rate limit exceeded.

    Args:
        response_json_string (str): The JSON string received from the GitHub API.

    Exits:
        SystemExit(1): If an unhandled API error (like rate limit exceeded) is found.
    """
    try:
        response_data = json.loads(response_json_string)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON response received:\n{response_json_string}", file=sys.stderr)
        sys.exit(1)

    message = response_data.get('message', '')

    if message:
        if "API rate limit exceeded for " in message:
            # The primary rate limit for unauthenticated requests is 60 requests per hour.
            # See https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting
            # TODO(xinxinmo): Write a short instruciton how to fix it.
            print(response_json_string, file=sys.stderr) # Print the original response to stderr
            print("Error: GitHub API rate limit exceeded.", file=sys.stderr)
            sys.exit(1) # Exit on critical error, matching shell script behavior
        # Add more `elif` checks for other specific messages if needed in the future
        # elif "Another error message" in message:
        #    print(f"Known error: {message}", file=sys.stderr)
        #    sys.exit(1)

def is_valid_commit(input_ref, org_repo):
    """
    Checks if an input string is a valid 40-character Git commit hash
    by attempting to fetch it from the GitHub API.

    Args:
        input_ref (str): The potential commit hash string.
        org_repo (str): The GitHub organization/repository string (e.g., "jax-ml/jax").

    Returns:
        bool: True if the input_ref is a valid and found commit hash, False otherwise.
    """
    # Check if input_ref looks like a 40-character hex string (commit SHA)
    if re.match(r"^[0-9a-fA-F]{40}$", input_ref):
        print("Input looks like a commit hash. Trying to fetch it to confirm it exists...")
        api_url = f"https://api.github.com/repos/{org_repo}/git/commits/{input_ref}"
        print(f"Checking if the first input is a commit SHA via API: {api_url}")

        try:
            # Use requests to perform the GET request
            response = requests.get(api_url, timeout=10) # Added timeout for robustness
            response_json_string = response.text # Get the raw response text

            # Call the helper to check for API messages like rate limits
            _check_github_rest_api_message(response_json_string)

            # Parse the JSON response
            response_data = response.json()
            sha = response_data.get('sha')

            # Check if SHA was found and is not null
            if not sha or sha == "null":
                print("Input is a 40-char hex string, but not a found commit. Please check your inputs.", file=sys.stderr)
                return False
            
            print("Input is a valid commit hash.")
            return True

        except requests.exceptions.RequestException as e:
            # Catch network errors, timeouts, etc.
            print(f"Error fetching commit details from GitHub API: {e}", file=sys.stderr)
            return False
        except json.JSONDecodeError:
            # This handles cases where _check_github_rest_api_message might not catch it,
            # or if the API returns non-JSON for an unexpected error.
            print(f"Error: Received non-JSON response for {api_url}:\n{response.text}", file=sys.stderr)
            return False
        except SystemExit:
            # This catches the sys.exit(1) from _check_github_rest_api_message,
            # allowing the main script to potentially handle it if this were a library call.
            # However, since the shell script exits, we just re-raise.
            raise
        except Exception as e:
            print(f"An unexpected error occurred during commit validation: {e}", file=sys.stderr)
            return False
    else:
        print("Input does not look like a 40-character commit hash. Please check your inputs.", file=sys.stderr)
        return False

def get_commit_hash_for_tag(tag_name, org_repo):
    """
    Gets the commit hash for a given tag name from a GitHub repository.
    If the tag is not found, it checks if the tag_name itself is a valid commit SHA.

    Args:
        tag_name (str): The name of the tag (e.g., "v1.0.0") or a potential commit SHA.
        org_repo (str): The GitHub organization/repository string (e.g., "octocat/Spoon-Knife").

    Returns:
        str: The commit hash if successfully found.

    Exits:
        SystemExit(1): If the commit hash cannot be determined.
    """
    commit_hash = None
    url = f"https://api.github.com/repos/{org_repo}/git/ref/tags/{tag_name}"

    print(f"Assume the first input is a tag. Attempting to get commit hash for '{tag_name}' from {url}.")

    try:
        response = requests.get(url, timeout=10)
        response_json_string = response.text
        _check_github_rest_api_message(response_json_string) # May cause sys.exit(1)

        response_data = response.json()
        # Access nested key: .object.sha
        object_data = response_data.get('object', {})
        commit_hash = object_data.get('sha')

    except requests.exceptions.RequestException as e:
        print(f"Warning: Error fetching tag '{tag_name}' from GitHub API: {e}", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Warning: Received non-JSON response for tag API call for '{tag_name}':\n{response.text}", file=sys.stderr)
    except SystemExit: # Catch and re-raise SystemExit from _check_github_rest_api_message
        raise
    except Exception as e:
        print(f"An unexpected error occurred during tag lookup: {e}", file=sys.stderr)

    if not commit_hash:
        print(f"Not an existing tag reference for '{tag_name}'. Trying to check if it is a commit SHA...")
        if is_valid_commit(tag_name, org_repo):
            commit_hash = tag_name # If it's a valid commit, use the input directly
        else:
            print(f"Error: Could not determine commit hash for ref '{tag_name}' under '{org_repo}'.", file=sys.stderr)
            sys.exit(1) # Exit if no commit hash can be found
    
    print(f"The commit hash is {commit_hash}.")
    return commit_hash

def download_seed_file_from_git_remote(org_repo, tag_or_commit, file_name):
    """Downloads a specified seed file from a GitHub repository.

    This function first determines the exact commit hash for a given tag or commit reference.
    It then constructs the raw GitHub content URL for the specified file within that commit
    and proceeds to download the file.

    Args:
        org_repo (str): The GitHub organization and repository name (e.g., "owner/repo_name").
        tag_or_commit (str): A Git tag name (e.g., "v1.0.0") or a full 40-character commit SHA.
        file_name (str): The path to the file within the repository at the given tag/commit
                         (e.g., "build/requirements_lock_3_10.txt").

    Raises:
        SystemExit: If an error occurs during the determination of the commit hash
                    (e.g., tag/commit not found, API rate limit exceeded) or
                    during the file download (e.g., file not found on remote, network error).
                    Errors are reported to stderr before exiting.
    """
    # Get the commit hash for the tag or directly validate if it's a commit
    COMMIT_HASH = get_commit_hash_for_tag(tag_or_commit, org_repo)

    final_seed_file_path = f"https://raw.githubusercontent.com/{org_repo}/{COMMIT_HASH}/{file_name}"

    # Download the python seed lock file
    download_remote_file(final_seed_file_path)

def fix_maxtext_requirements(file_path: str):
    """
    Fixes specific dependency pins in a MaxText-related requirements.txt file.

    This function addresses the following issues:
    - Unpins protobuf from 3.20.3 (as it's outdated and better managed by TensorFlow).
    - Changes sentencepiece to allow newer versions (>=0.1.97).
    - Pins specific source links (JetStream and logging) to a known commit hash
      to ensure reproducibility.

    Args:
        file_path (str): The path to the requirements.txt file to modify.
                         Defaults to "requirements.txt" in the current directory.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    replacements = {
        "protobuf==3.20.3": "protobuf",
        "sentencepiece==0.1.97": "sentencepiece>=0.1.97",
        "/JetStream.git": "/JetStream.git@261f25007e4d12bb57cf8d5d61e291ba8f18430f",
        "/logging.git": "/logging.git@44b4810e65e8c0a7d9e4e207c60e51d9458a3fb8",
    }

    try:
        with fileinput.FileInput(file_path, inplace=True) as f:
            for line in f:
                new_line = line
                for old_str, new_str in replacements.items():
                    new_line = new_line.replace(old_str, new_str)
                print(new_line, end='')
        print(f"Successfully updated '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# TODO: Create tests and remove the following examples.
# Example usage:
if __name__ == "__main__":
    # ORG_REPO = "jax-ml/jax" # A public repo for testing

    # print("\n--- Test Case 1: Valid Tag ---")
    # # A known good tag from octocat/Spoon-Knife
    # tag_name_valid = "jax-v0.6.2"
    # try:
    #     result_hash = get_commit_hash_for_tag(tag_name_valid, ORG_REPO)
    #     print(f"Resolved hash for '{tag_name_valid}': {result_hash} (Test PASSED).")
    # except SystemExit as e:
    #     print(f"Test 1 FAILED: Script exited with code {e.code}.")

    # print("\n--- Test Case 2: Non-existent Tag, but Valid Commit SHA ---")
    # commit_sha_from_tag = "1ad05bb26105f23ee7728b36cca12901fe70e187"
    # try:
    #     result_hash = get_commit_hash_for_tag(commit_sha_from_tag, ORG_REPO)
    #     print(f"Resolved hash for '{commit_sha_from_tag}' (as SHA): {result_hash} (Test PASSED).")
    # except SystemExit as e:
    #     print(f"Test 2 FAILED: Script exited with code {e.code}.")

    # print("\n--- Test Case 3: Non-existent Tag and Invalid Commit SHA ---")
    # non_existent_ref = "non-existent-tag-or-sha-1234"
    # try:
    #     get_commit_hash_for_tag(non_existent_ref, ORG_REPO)
    #     print(f"Test 3 FAILED: Expected script to exit for '{non_existent_ref}'.")
    # except SystemExit as e:
    #     if e.code == 1:
    #         print(f"Test 3 PASSED: Script exited with code {e.code} as expected for '{non_existent_ref}'.")
    #     else:
    #         print(f"Test 3 FAILED: Script exited with unexpected code {e.code} for '{non_existent_ref}'.")

 
    # print("\n--- Test Case 1: Valid Commit Hash ---")
    # # A known good commit from octocat/Spoon-Knife
    # valid_sha = "1ad05bb26105f23ee7728b36cca12901fe70e187"
    # if is_valid_commit(valid_sha, "jax-ml/jax"):
    #     print(f"'{valid_sha}' is a valid commit. Test PASSED.")
    # else:
    #     print(f"'{valid_sha}' is NOT a valid commit. Test FAILED.")

    # print("\n--- Test Case 2: Non-existent Commit Hash (valid format) ---")
    # # A randomly generated 40-char hex string, unlikely to exist
    # non_existent_sha = "1234567890abcdef1234567890abcdef12345678"
    # if not is_valid_commit(non_existent_sha, ORG_REPO):
    #     print(f"'{non_existent_sha}' is correctly identified as not found. Test PASSED.")
    # else:
    #     print(f"'{non_existent_sha}' incorrectly identified as valid. Test FAILED.")

    # print("\n--- Test Case 3: Invalid Format (too short) ---")
    # invalid_format_short = "12345"
    # if not is_valid_commit(invalid_format_short, ORG_REPO):
    #     print(f"'{invalid_format_short}' is correctly identified as invalid format. Test PASSED.")
    # else:
    #     print(f"'{invalid_format_short}' incorrectly identified as valid. Test FAILED.")

    # generate_pyproject_toml("3.11")
    # if not os.path.exists("uv.lock"):
    #     with open("uv.lock", "w") as f: f.write("")
    # with open("seed_requirements.txt", "w") as f:
    #     f.write("numpy==1.26.4\n")
    #     f.write("pandas==2.2.2\n")
    # with open("project_requirements.txt", "w") as f:
    #     f.write("requests==2.31.0\n")
    # # Run the function
    # # Note: For this to work, 'uv' must be installed and accessible in your system's PATH.
    # status = build_seed_env(
    #     seed_file="seed_requirements.txt",
    #     project_requirements_file="project_requirements.txt",
    #     output_file="uv_lock_output.txt"
    # )
    # print(f"\nbuild_seed_env finished with status: {status}")
    # # Clean up dummy files
    # for f in ["uv.lock", "seed_requirements.txt", "project_requirements.txt",
    #          "uv_lock_output.txt"]:
    #     if os.path.exists(f):
    #         os.remove(f)
  

    # print("--- Test Case 1: Successful generation (default output) ---")
    # generate_pyproject_toml("3.10")
    # print("-" * 50)
    # # Clean up for next test
    # if os.path.exists("./pyproject.toml"):
    #     os.remove("./pyproject.toml")
    # # 2. Successful generation with a specified output file
    # print("--- Test Case 2: Successful generation (custom output) ---")
    # generate_pyproject_toml("3.11", "./my_project/pyproject.toml")
    # print("-" * 50)
    # # Clean up for next test
    # if os.path.exists("./my_project/pyproject.toml"):
    #     os.remove("./my_project/pyproject.toml")
    #     os.rmdir("./my_project") # Remove the directory if it was created
    # # 3. Error case: Missing Python version
    # print("--- Test Case 3: Error - Missing Python version ---")
    # generate_pyproject_toml("")
    # print("-" * 50)
    # # 4. Error case: Invalid Python version format
    # print("--- Test Case 4: Warning - Invalid Python version format ---")
    # generate_pyproject_toml("3")
    # print("-" * 50)
    # # 5. Error case: Another invalid Python version format
    # print("--- Test Case 5: Warning - Another invalid Python version format ---")
    # generate_pyproject_toml("3.10.5")
    # print("-" * 50)

    # A valid url
    # test_url = "https://raw.githubusercontent.com/jax-ml/jax/1ad05bb26105f23ee7728b36cca12901fe70e187/build/requirements_lock_3_12.txt"
    # Invalid url
    # test_url = "https://example.com/nonexistent_file.txt" # Example 404 URL
    # download_remote_file(test_url)

    pass
