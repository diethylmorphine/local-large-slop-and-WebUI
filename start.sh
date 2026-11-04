#!/usr/bin/env bash
set -eu

requirements_without_llama=$(mktemp)
trap 'rm -f "$requirements_without_llama"' EXIT

grep -vE '^[[:space:]]*llama-cpp-python([<>=!~;[:space:]]|$)' requirements.txt \
    > "$requirements_without_llama"

python -m pip install \
    --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    --only-binary=llama-cpp-python \
    llama-cpp-python\>=0.3.8\,\<0.4.0
python -m pip install --no-cache-dir -r "$requirements_without_llama"

exec python cpu_limit.py
