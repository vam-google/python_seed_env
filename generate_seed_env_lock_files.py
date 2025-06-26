import os
import subprocess
import sys

def main(seed_repo_lock_file,
         host_repo_requirements_file,
         constraints_gpu_only_file,
         constraints_tpu_only_file,
         final_output_file,
         building_gpu_env=False,
         building_tpu_env=False):
    """
    Manages uv dependencies based on provided file paths and environment flags.

    Args:
        seed_repo_lock_file (str): Path to the seed repository's requirements_lock.txt file.
        host_repo_requirements_file (str): Path to the host repository's requirements.txt file.
        constraints_gpu_only_file (str): Path to the constraints_gpu_only.txt file.
        constraints_tpu_only_file (str): Path to the constraints_tpu_only.txt file.
        final_output_file (str): Path to the output lock file for host repository.
        building_gpu_env (bool): True if building a GPU environment, False otherwise.
        building_tpu_env (bool): True if building a TPU environment, False otherwise.
    """
    print("Starting generating seed environment lock files for host repository.")

    # 1. Add the packages listed in requirements_lock_3_12.txt to the project's dependencies
    # `uv add --managed-python --no-build --no-sync --resolution=highest -r requirements_lock_3_12.txt`
    print(f"\nAdding {seed_repo_lock_file}.")
    run_uv_command([
        "add",
        "--managed-python",
        "--no-build",
        "--no-sync",
        "--resolution=highest",
        "-r", seed_repo_lock_file
    ])
    print(f"\nAdded {seed_repo_lock_file}.")

    # 2. Remove unnecessary dependencies
    if not building_gpu_env:
        # Remove GPU dependencies
        # `cat constraints_gpu_only.txt | xargs -I {} uv remove --managed-python --no-sync --resolution=highest {}`
        print(f"\nConditionally removing GPU-only constraints.")
        gpu_packages_to_remove = get_packages_from_file(constraints_gpu_only_file)
        if gpu_packages_to_remove:
            for package in gpu_packages_to_remove:
                run_uv_command([
                    "remove",
                    "--managed-python",
                    "--no-sync",
                    "--resolution=highest",
                    package
                ])
        else:
            print(f"No GPU-only packages found in {constraints_gpu_only_file} or file does not exist.")
    else:
        print("\nBuilding GPU env, not removing GPU-only constraints.")

    if not building_tpu_env:
        # Remove TPU dependencies
        # `cat constraints_tpu_only.txt | xargs -I {} uv remove --managed-python --no-sync --resolution=highest {}`
        print(f"\nConditionally removing TPU-only constraints.")
        tpu_packages_to_remove = get_packages_from_file(constraints_tpu_only_file)
        if tpu_packages_to_remove:
            for package in tpu_packages_to_remove:
                run_uv_command([
                    "remove",
                    "--managed-python",
                    "--no-sync",
                    "--resolution=highest",
                    package
                ])
        else:
            print(f"No TPU-only packages found in {constraints_tpu_only_file} or file does not exist.")
    else:
        print("\nBuilding TPU env, not removing TPU-only constraints.")

    # 3. Resolving the dependency version conflicts between project.toml and maxtext_requirement.txt
    #    by lower-bounding the dependencies.
    
    # `uv add --managed-python --no-sync --resolution=highest -r requirements.txt`
    run_uv_command([
        "add",
        "--managed-python",
        "--no-sync",
        "--resolution=highest",
        "-r", host_repo_requirements_file
    ])

    # `uv export --managed-python --locked --no-hashes --no-annotate --resolution=highest --output-file=maxtext_requirements_lock_3_12.txt`
    run_uv_command([
        "export",
        "--managed-python",
        "--locked",
        "--no-hashes",
        "--no-annotate",
        "--resolution=highest",
        "--output-file", final_output_file
    ])

    print("\nuv dependency management completed.")

def run_uv_command(command_parts, check_error=True):
    """
    Runs a uv command using subprocess.

    Args:
        command_parts (list): A list of strings representing the command and its arguments.
        check_error (bool): If True, raises an exception if the command returns a non-zero exit code.
    
    Returns:
        subprocess.CompletedProcess: The result of the subprocess run.
    """
    full_command = [sys.executable, "-m", "uv"] + command_parts
    print(f"Running command: {' '.join(full_command)}")
    try:
        result = subprocess.run(full_command, capture_output=True, text=True, check=check_error)
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: 'uv' command not found. Please ensure 'uv' is installed and in your PATH.")
        print(f"You can install it using: pip install uv")
        raise

def get_packages_from_file(file_path):
    """Reads package names from a file, one per line."""
    packages = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'): # Ignore empty lines and comments
                    packages.append(line)
    return packages
