# POC for creating reproducible jax-based python environments for MaxText

## Some non-obvious notes
1. The general idea of the approach is to rely on already vast and well tested locked set of deps that JAX itself uses for a big chunk of its own testing (including presubmit and continuous jobs). Propagating those dependencies inside MaxText environment ensures that for a specific JAX version MaxText runs itself on as close environment as possible to what JAX of the same version was testing itself when it was getting released.
2. To get an idea how this all works under the hood check the `build_seed_env.sh`, it is pretty short and self explanatory for the most part.
3. CUDA deps are pulled as python wheels, which is the recommended for JAX to get CUDA, no system-wide cuda packages are needed except driver.
4. Presense of libtpu in an env makes jax to assume that it  must run on TPU, so for any GPU-based workflows libtpu must be excluded (thus the `constraints_tpu_only.txt` file)
5. CUDA wheels are big and heavy, installing for TPU workflows is an unnecessary waste of resources (thus the `constraints_gpu_only.txt`).  

## Quick start
0. Install [uv](https://docs.astral.sh/uv/).
1. Always start in a directory with minimal `pyproject.toml` (as it is in this repo), and no `uv.lock` file present.
2. Run `./build_seed_env.sh`
3. The script above will produce `maxtext_requirements_lock_3_12.txt` which will contain a full set of locked maxtext python dependencies pinned to the highest version numbers available when you ran it. 
4. Use `maxtext_requirements_lock_3_12.txt` it to set up any virtual env or Docker container you want to run MaxText in.
5. Re-running `./build_seed_env.sh` at any future point in time is non-reproducible.
6. The script above also produces a `pyproject.toml`, which lists same dependencies as in the lock.txt but in a lower-bound form.
7. The `pyproject.toml` should be comitted in source tree every time it is updated (see step #1).
8. If `pyproject.toml` is comitted, running `uv export --managed-python --locked --no-hashes --no-annotate --resolution=lowest --output-file=maxtext_requirements_lock_3_12.txt` on that commit at any point in time in the future is reproducible.
9. MaxText may have different `pyproject.toml` (in different folders), each corresponding to a specific workflow.
10. For any commit in MaxText (assuming `pyproject.toml` is checked in), use command #8 to recreate MaxText Python environment for that commit. 
11. To generate `pyproject.toml` and `requirements_lock.txt` for a different python version change `requires-python` line in `pyproject.toml` and pull matching jax `requirements_lock_<py_ver>.txt` in `build_seed_env.sh` repeat process from scratch (pyproject.toml should be with no deps and no `uv.lock` file should be present).
12. TBD: Use pyproject.toml to generate MaxText meta wheel, with all its deps lower-bounded, but not pinned.
