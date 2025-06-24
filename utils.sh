# Helper functions for seed preparation

download_remote_file() {
    local file_url="$1"

    # Checking existence of the remote file. Perform a HEAD request to get only the HTTP status code.
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --head "$file_url")
    local curl_exit_status=$?
    if [ "$curl_exit_status" -ne 0 ]; then
        echo "Error: curl command failed with exit status $curl_exit_status for existence check of $file_url." >&2
        exit 1 # Exit on network/curl error
    fi
    if [ "$http_code" -eq 404 ]; then
        echo "Error: File not found (HTTP 404) at $file_url."
        exit 1
    elif [ "$http_code" -ne 200 ]; then
        echo "Error: Unexpected HTTP status code $http_code for $file_url. Expected 200 OK."
        exit 1 # Exit for other non-200, non-404 issues
    fi

    # Download the file
    curl "$file_url" -O
}

generate_pyproject_toml() {
  local python_version="$1"
  local output_file="${2:-./pyproject.toml}" # Default to ./pyproject.toml if $2 is empty

  if [ -z "$python_version" ]; then
    echo "Error: Python version is required."
    echo "Usage: generate_pyproject_toml <python_version> [output_file]"
    exit 1
  fi

  # Basic validation for python_version format (e.g., "3.10", not just "3")
  if ! [[ "$python_version" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Warning: Python version '$python_version' does not seem to be in 'X.Y' format."
  fi

  cat << EOF > "$output_file"
[project]
name = "python_seed_env"
version = "0.1.0"
requires-python = "==${python_version}.*"
dependencies = [
]
EOF

  if [ $? -eq 0 ]; then
    echo "Successfully generated '$output_file' for Python version '$python_version'."
    return 0
  else
    echo "Error: Failed to generate '$output_file'." >&2
    return 1
  fi
}

build_seed_env() {
    local seed="$1"
    local project_requirements="$2"
    local output_file="$3"
    # Commands to create env
    # ---------------------------------
    # Remove uv.lock if one exists
    rm -f uv.lock
    uv add --managed-python --no-build --no-sync --resolution=highest -r "$seed"
    # Uncomment if building GPU env
    # cat constraints_gpu_only.txt | xargs -I {} uv remove --managed-python --no-sync --resolution=highest {}
    # Comment if building TPU env
    ###################### TODO: Add an argument to exclude contraint deps #################################
    cat constraints_tpu_only.txt | xargs -I {} uv remove --managed-python --no-sync --resolution=highest {}
    # Here if there are any deps in project.toml that conflict with
    # maxtext_requirements.txt, lower bound them  in project.toml manually
    uv add --managed-python --no-sync --resolution=highest -r "$project_requirements"
    uv export --managed-python --locked --no-hashes --no-annotate --resolution=highest --output-file="$output_file"
    python3 lock_to_lower_bound_project.py "$output_file" pyproject.toml
    rm uv.lock
    uv lock --managed-python --resolution=lowest
    uv export --managed-python --locked --no-hashes --no-annotate --resolution=lowest --output-file="$output_file"
}

_check_github_rest_api_message() {
    local RESPONSE="$1"
    local message=$(echo "$RESPONSE" | jq -r '.message // empty')
    if [ -n "$message" ]; then
        if [[ "$message" =~ "API rate limit exceeded for " ]]; then
            # The primary rate limit for unauthenticated requests is 60 requests per hour.
            # See https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting
            # TODO(xinxinmo): Find a way to fix it.
            echo "$RESPONSE"
            exit 1
        fi
    fi
}

is_valid_commit() {
    local input_ref="$1"
    local org_repo="$2"
    local api_url
    local RESPONSE
    local SHA

    if [[ "$input_ref" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "Input looks like a commit hash. Try to fetch it to confirm it exists..."
        api_url="https://api.github.com/repos/${org_repo}/git/commits/${input_ref}"
        echo "Checking if the first input is a commit SHA via API: $api_url"

        RESPONSE=$(curl -s "$api_url")
        _check_github_rest_api_message "$RESPONSE"

        SHA=$(echo "$RESPONSE" | jq -r '.sha')
        if [ -z "$SHA" ] || [ "$SHA" == "null" ]; then
            echo "Input is a 40-char hex string, but not a found commit. Please check your inputs."
            return 1 # False (failure)
        fi
        echo "Input is a valid commit hash."
        return 0 # True (success)
    fi
    echo "Input does not look like a commit hash. Please check your inputs."
    return 1
}

# Function to get the commit hash for a given tag
get_commit_hash_for_tag() {
    local tag_name="$1"
    local org_repo="$2"
    local url="https://api.github.com/repos/${org_repo}/git/ref/tags/${tag_name}"
    local RESPONSE

    echo "Assume the first input is a tag. Attempting to get commit hash for it."
    echo "API URL: $url"

    RESPONSE=$(curl -s "$url")
    _check_github_rest_api_message "$RESPONSE"

    export COMMIT_HASH=$(echo "$RESPONSE" | jq -r '.object.sha')

    if [ -z "$COMMIT_HASH" ] || [ "$COMMIT_HASH" == "null" ]; then
        echo "Not an existing tag reference. Trying to check if it is a commit SHA..."
        if is_valid_commit "$tag_name" "$org_repo"; then
            export COMMIT_HASH=$tag_name
        else
            echo "Error: Could not determine commit hash for ref '$tag_name' under '$org_repo'."
            exit 1
        fi
    fi

    echo "The commit hash is $COMMIT_HASH."
}
