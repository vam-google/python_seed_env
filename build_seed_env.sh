#!/bin/bash
set -x
set -e

# Picking latest (for the moment of writing this) released JAX and latest
# MaxText commit
curl https://raw.githubusercontent.com/jax-ml/jax/c09b1bb763d846a694f919e5a5adda9575ce66d6/build/requirements_lock_3_12.txt -O
curl https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/80c0884a413219bbc317eaea20571d995cbe6566/requirements.txt -O

# Fixing current maxtext requirements.txt
# For whatever reason MaxText pins protobuf dependency to 3.20.3 which is not even supported anymore
# Note, it is better to let tensorflow (which is a dependency here) to define protobuf dep)
sed -i 's/protobuf==3.20.3/protobuf/g' requirements.txt
sed -i 's/sentencepiece==0.1.97/sentencepiece>=0.1.97/g' requirements.txt
# All source links must be pinned to a hash, there is no other way to guarantee reproducibility
# for source links. They are a very small minority of deps, so imposing this requirement should be fine
sed -i 's/\/JetStream.git/\/JetStream.git@261f25007e4d12bb57cf8d5d61e291ba8f18430f/g' requirements.txt
sed -i 's/\/logging.git/\/logging.git@44b4810e65e8c0a7d9e4e207c60e51d9458a3fb8/g' requirements.txt

# Commands to create env
# ---------------------------------
uv add --managed-python --no-build --no-sync --resolution=highest -r requirements_lock_3_12.txt
# Comment if building GPU env
cat constraints_gpu_only.txt | xargs -I {} uv remove --managed-python --no-sync --resolution=highest {}
# Comment if building TPU env
# cat constraints_tpu_only.txt | xargs -I {} uv remove --managed-python --no-sync --resolution=highest {}
# Here if there are any deps in project.toml that conflict with
# maxtext_requirements.txt, lower bound them  in project.toml manually
uv add --managed-python --no-sync --resolution=highest -r requirements.txt
uv export --managed-python --locked --no-hashes --no-annotate --resolution=highest --output-file=./tpu_seed/python_3_12/maxtext_requirements_lock_3_12.txt
python3 lock_to_lower_bound_project.py ./tpu_seed/python_3_12/maxtext_requirements_lock_3_12.txt pyproject.toml
rm uv.lock
uv lock --managed-python --resolution=lowest
uv export --managed-python --locked --no-hashes --no-annotate --resolution=lowest --output-file=./tpu_seed/python_3_12/maxtext_requirements_lock_3_12.txt

# Congrats, you've done it:
#   - maxtext_requirements_lock_3_12.txt is your well-defined reproducible python
#     environment to install in a virtual env or a Docker image
#   - pyproject.toml - commit this into your github repo, using it one will
#     always be able to reproduce the environment from maxtext commit by running:
#       uv lock --managed-python --resolution=lowest # recreates uv.lock
#       uv export --managed-python --locked --no-hashes --no-annotate --resolution=lowest --output-file=maxtext_requirements_lock_3_12.txt # regenerates lock.txt


# Commands to validate and inspect created env
# ---------------------------------
# uv sync --managed-python --locked --resolution=lowest
# uv cache clean
# uv tree --resolution=lowest --locked
# uv venv --managed-python --seed

