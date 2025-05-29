import sys
import re

def replace_dependencies_in_project_toml(new_deps, filepath = "pyproject.toml"):
  dependencies_regex = re.compile(
    r"^dependencies\s*=\s*\[(\n+\s*.*,\s*)*[\n\r]*\]",
    re.MULTILINE
  )

  with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
  new_content = dependencies_regex.sub(new_deps, content)

  with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)


def read_requirements_lock_file(filepath):
  lines = []
  with open(filepath, 'r', encoding='utf-8') as file:
    for line in file:
      if "#" not in line and ("==" in line or "@" in line):
        lines.append(line.strip())
  return lines

def convert_deps_to_lower_bound(pinned_deps):
  lower_bound_deps = []
  for pinned_dep in pinned_deps:
    lower_bound_dep = pinned_dep
    if "==" in pinned_dep:
      lower_bound_dep = pinned_dep.replace("==", ">=")
    lower_bound_deps.append(lower_bound_dep)

  return lower_bound_deps

def lower_boud_deps_to_string(lower_bound_deps):
  return 'dependencies = [\n    "' + '",\n    "'.join(lower_bound_deps) + '"\n]'

if __name__ == "__main__":
  pinned_deps = read_requirements_lock_file(sys.argv[1])
  lower_bound_deps = convert_deps_to_lower_bound(pinned_deps)
  new_deps = lower_boud_deps_to_string(lower_bound_deps)
  replace_dependencies_in_project_toml(new_deps, sys.argv[2])