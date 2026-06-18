#!/usr/bin/env bash
# CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 ./sglang-start-qwen3-8b.sh
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/data/model/llama-3-8b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-16000}"
DTYPE="${DTYPE:-bfloat16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.40}"
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
GPU_ID="${GPU_ID:-0}"

# Keep the dedup hook in front of the runtime search path, but make sure
# FlashInfer sees a clean CUDA toolkit path when it looks for nvcc.
unset CUDA_HOME
unset CUDA_PATH
unset CUDACXX
unset NVCC
export CUDA_HOME=/root/dedup/build/cuda
export PATH=/root/dedup/build/cuda/bin:/usr/local/cuda-12.4/bin:${PATH}

# Let SGLang JIT use a newer host compiler if available, without changing the
# system default toolchain used by dedup.
if [[ -x /usr/bin/gcc-11 && -x /usr/bin/g++-11 ]]; then
  export CC="${CC:-/usr/bin/gcc-11}"
  export CXX="${CXX:-/usr/bin/g++-11}"
else
  export CC="${CC:-gcc}"
  export CXX="${CXX:-g++}"
fi
# nvcc chooses its own host compiler unless CUDAHOSTCXX is set explicitly.
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
export CUDACXX=/root/dedup/build/cuda/bin/nvcc
export NVCC=/root/dedup/build/cuda/bin/nvcc

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --load-format "${LOAD_FORMAT}" \
  --trust-remote-code \
  --disable-piecewise-cuda-graph 
