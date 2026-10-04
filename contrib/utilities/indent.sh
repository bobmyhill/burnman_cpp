#!/usr/bin/env bash
# Format repository sources, including untracked files that are not ignored.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
cd -- "$repo_root"

# Check both formatters before changing any files.
for formatter in clang-format black; do
  if ! command -v "$formatter" >/dev/null 2>&1; then
    printf 'Required formatter not found on PATH: %s\n' "$formatter" >&2
    exit 1
  fi
  "$formatter" --version >/dev/null
done

cpp_files=()
py_files=()
while IFS= read -r -d '' file; do
  [[ -f "$file" ]] || continue
  case "$file" in
    *.cpp|*.hpp) cpp_files+=("$file") ;;
    *.py) py_files+=("$file") ;;
  esac
done < <(git ls-files -z --cached --others --exclude-standard -- '*.cpp' '*.hpp' '*.py')

if ((${#cpp_files[@]})); then
  clang-format -i --style=file -- "${cpp_files[@]}"
fi
if ((${#py_files[@]})); then
  black -- "${py_files[@]}"
fi
