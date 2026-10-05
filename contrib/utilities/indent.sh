#!/usr/bin/env bash
# Format repository sources, including untracked files that are not ignored.
set -euo pipefail

usage() {
  printf 'Usage: %s [--check] [--tracked]\n' "${0##*/}"
  printf '  --check    Report formatting problems without changing files.\n'
  printf '  --tracked  Only include tracked files (the GitHub check uses this).\n'
}

check=false
git_file_options=(--cached --others --exclude-standard)
for option in "$@"; do
  case "$option" in
    --check) check=true ;;
    --tracked) git_file_options=(--cached) ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$option" >&2; usage >&2; exit 2 ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
cd -- "$repo_root"

# Check both formatters and their versions before changing any files.
while IFS= read -r requirement || [[ -n "$requirement" ]]; do
  [[ -z "$requirement" || "$requirement" == \#* ]] && continue
  formatter="${requirement%%==*}"
  required_version="${requirement#*==}"
  if ! command -v "$formatter" >/dev/null 2>&1; then
    printf 'Required formatter not found on PATH: %s\n' "$formatter" >&2
    printf 'Install with: python -m pip install -r contrib/utilities/requirements-format.txt\n' >&2
    exit 1
  fi
  version_output="$("$formatter" --version)"
  if [[ ! "$version_output" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]] ||
     [[ "${BASH_REMATCH[1]}" != "$required_version" ]]; then
    printf 'Required %s version: %s; found: %s\n' "$formatter" "$required_version" "$version_output" >&2
    printf 'Install with: python -m pip install -r contrib/utilities/requirements-format.txt\n' >&2
    exit 1
  fi
done < "$script_dir/requirements-format.txt"

cpp_files=()
py_files=()
while IFS= read -r -d '' file; do
  [[ -f "$file" ]] || continue
  case "$file" in
    *.cpp|*.hpp) cpp_files+=("$file") ;;
    *.py) py_files+=("$file") ;;
  esac
done < <(git ls-files -z "${git_file_options[@]}" -- '*.cpp' '*.hpp' '*.py')

if "$check"; then
  formatting_status=0
  if ((${#cpp_files[@]})); then
    clang-format --dry-run --Werror --style="file:$repo_root/.clang-format" -- "${cpp_files[@]}" || formatting_status=1
  fi
  if ((${#py_files[@]})); then
    black --check --diff --config "$repo_root/pyproject.toml" -- "${py_files[@]}" || formatting_status=1
  fi
  exit "$formatting_status"
else
  if ((${#cpp_files[@]})); then
    clang-format -i --style="file:$repo_root/.clang-format" -- "${cpp_files[@]}"
  fi
  if ((${#py_files[@]})); then
    black --config "$repo_root/pyproject.toml" -- "${py_files[@]}"
  fi
fi
