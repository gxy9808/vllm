#!/usr/bin/env bash
# Regenerate compact Python locks consumed by ROCm base and CI images.

set -euo pipefail

UV_VERSION="0.11.1"
uv_command=(uv)
if [[ "$(uv --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
    uv_command=(uvx --from "uv==${UV_VERSION}" uv)
fi

excluded_packages=(
    torch torchvision torchaudio triton
    cuda-bindings cuda-pathfinder cuda-toolkit cupy-cuda12x
)
for suffix in "" -cu12 -cu13; do
    for package in \
        nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
        nvidia-cuda-runtime nvidia-cudnn nvidia-cufft nvidia-cufile \
        nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt \
        nvidia-nccl nvidia-nvjitlink nvidia-nvshmem nvidia-nvtx; do
        excluded_packages+=("${package}${suffix}")
    done
done

exclude_args=()
for package in "${excluded_packages[@]}"; do
    exclude_args+=(--no-emit-package "${package}")
done

tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT
common_args=(
    --quiet
    --index-strategy unsafe-best-match
    --python-platform x86_64-manylinux_2_28
    --python-version 3.12
    --no-annotate
    --no-header
    "${exclude_args[@]}"
)

compile_lock() {
    local input=$1
    local output=$2
    shift 2

    rm -f "${tmp_dir}/resolved.txt"
    [[ ! -f "${output}" ]] || cp "${output}" "${tmp_dir}/resolved.txt"
    "${uv_command[@]}" pip compile "${input}" --output-file "${tmp_dir}/resolved.txt" \
        "${common_args[@]}" "$@"
    sed -n '/^# The following packages were excluded/q; /^$/d; p' \
        "${tmp_dir}/resolved.txt" > "${output}"
}

compile_lock requirements/rocm-ci.in requirements/rocm-ci.txt \
    --constraint requirements/test/rocm.txt

if [[ -f requirements/rocm-lmcache.txt ]]; then
    cp requirements/rocm-lmcache.txt "${tmp_dir}/lmcache-full.txt"
fi
compile_lock requirements/rocm-lmcache.in "${tmp_dir}/lmcache-full.txt" \
    --constraint requirements/rocm-ci.txt \
    --constraint requirements/test/rocm.txt

awk -F '==' '/^[A-Za-z0-9][A-Za-z0-9._-]*==/ {
    name = tolower($1); gsub(/[-_.]+/, "-", name); print name
}' requirements/rocm-ci.txt requirements/test/rocm.txt \
    | sort -u > "${tmp_dir}/baseline.txt"
awk -F '==' 'NR == FNR { seen[$1] = 1; next }
/^[A-Za-z0-9][A-Za-z0-9._-]*==/ {
    name = tolower($1); gsub(/[-_.]+/, "-", name)
    if (!(name in seen)) print
}' "${tmp_dir}/baseline.txt" "${tmp_dir}/lmcache-full.txt" \
    > requirements/rocm-lmcache.txt
