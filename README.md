[![CI](https://github.com/dapovoa/CTranslate2/workflows/CI/badge.svg)](https://github.com/dapovoa/CTranslate2/actions?query=workflow%3ACI) [![Release](https://img.shields.io/github/v/release/dapovoa/CTranslate2)](https://github.com/dapovoa/CTranslate2/releases/latest) [![ROCm](https://img.shields.io/badge/ROCm-6.3.1-red.svg)](BUILD_ROCM.md) [![CTranslate2](https://img.shields.io/badge/CTranslate2-v4.6.0-blue.svg)](https://github.com/OpenNMT/CTranslate2) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# CTranslate2 - AMD ROCm Fork

This is a fork of [ROCm/CTranslate2](https://github.com/ROCm/CTranslate2) (AMD's official ROCm port of [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2)), providing upgraded and optimized Transformer model inference on AMD GPUs.

**Latest Release:** [v4.6.0-rocm6.3.1](https://github.com/dapovoa/CTranslate2/releases/tag/v4.6.0-rocm6.3.1) | **Upstream:** [ROCm/CTranslate2](https://github.com/ROCm/CTranslate2)

---

## What This Fork Provides

The official ROCm fork (ROCm/CTranslate2) is currently at v3.23.0 and targets ROCm 6.1. This fork brings several important improvements:

### Core Upgrades

**CTranslate2 v4.6.0**
- Upgraded from v3.23.0 to v4.6.0, bringing 70+ upstream commits
- Includes support for newer models: Gemma 2, Qwen2, Phi-3, Mistral Nemo
- Flash Attention support for faster inference
- AWQ quantization (INT4) for smaller models
- Tensor parallelism for multi-GPU inference

### ROCm Compatibility

**Modern ROCm Support (6.2+, 6.3+, 7.0+)**
- Updated HIP API calls for ROCm 6.3+ compatibility
- Automatic hipBLAS API version detection at compile time
- Fixed include paths for ROCm 6.3 header reorganization
- Tested on ROCm 6.2.4, 6.3.0, 6.3.1, and 7.0.1

### Whisper and Speech Models

**MIOpen Integration**
- Properly links MIOpen (AMD's cuDNN equivalent)
- Fixes the common "Conv1D on GPU currently requires the cuDNN library" error
- Full Conv1D support for Whisper, Wav2Vec2, and Wav2Vec2Bert models
- Compatible with `faster-whisper` Python package

### Build Improvements

**GPU-Only Build Configuration**
- Disabled CPU dispatch kernels (`ENABLE_CPU_DISPATCH=OFF`)
- Removed CPU backend dependencies (DNNL, OpenBLAS)
- Cleaner GPU-focused implementation
- Automated verification checks in build script

**Automated Build Script**
- `build_rocm.sh` handles entire build process
- Auto-detects GPU architecture (gfx1100, etc.)
- Builds both C++ library and Python wheel
- Includes CMake cache verification and symbol checks

---

## Quick Start

### Installation from Pre-Built Wheel

Download the latest wheel from [Releases](https://github.com/dapovoa/CTranslate2/releases):

```bash
pip install ctranslate2-4.6.0+rocm6.3.1-cp312-cp312-linux_x86_64.whl
```

**System Requirements:**
- ROCm 6.3.1 (ROCm 6.2+ and 7.0+ also supported)
- AMD Radeon RX 7900 XTX or compatible RDNA3 GPU (gfx1100)
- Python 3.12
- Ubuntu 24.04 LTS (or compatible Linux distribution)

### Build from Source

```bash
# Clone with submodules
git clone --branch amd_dev_v4.6.0_rocm6.3 --recursive https://github.com/dapovoa/CTranslate2.git
cd CTranslate2

# Run automated build script
./build_rocm.sh

# Python wheel will be in python/dist/
```

For detailed build instructions and manual build steps, see [BUILD_ROCM.md](BUILD_ROCM.md).

---

## Technical Details

### Build Configuration

This fork uses the following CMake configuration:

```cmake
-DCMAKE_BUILD_TYPE=Release
-DWITH_CUDA=ON                    # HIP uses CUDA flag
-DWITH_CUDNN=ON                   # MIOpen uses CUDNN flag
-DWITH_MKL=OFF
-DWITH_DNNL=OFF
-DWITH_OPENBLAS=OFF
-DOPENMP_RUNTIME=NONE
-DENABLE_CPU_DISPATCH=OFF         # GPU-only build
-DGPU_RUNTIME=HIP
-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
-DAMDGPU_TARGETS=gfx1100
```

### Key Implementation Changes

**CUDA to HIP Translation**
- Custom header files for CUDA→HIP compatibility:
  - `src/cuda2hip_macros.hpp` - CUDA/HIP macro translations
  - `src/cuda2hip_types.hpp` - C++ type compatibility layer
  - `src/cuda2hip_device.hpp` - Device function mappings
- CUB→hipCUB and cuRAND→hipRAND namespace mappings
- Thrust device system set to `THRUST_DEVICE_SYSTEM_HIP`

**ROCm 6.3+ Compatibility**
- Updated hipBLAS include paths (`hipblas/hipblas.h`)
- Added ROCm include directories to `CMAKE_CXX_FLAGS`
- ROCPrim type traits fixes for ROCm 6.3
- Conditional compilation based on hipBLAS version

**MIOpen Conv1D**
- Proper MIOpen library linking in CMakeLists.txt
- Conv1D GPU kernel implementation for Whisper models
- Tested with faster-whisper on real-time audio streams

---

## Tested Configurations

**Primary Configuration:**
- ROCm 6.3.1
- AMD Radeon RX 7900 XTX (gfx1100)
- Ubuntu 24.04 LTS
- Python 3.12

**Also Tested:**
- ROCm 6.2.4 on AMD Radeon RX 7900 XT (gfx1100)
- ROCm 7.0.1 on AMD Radeon RX 7900 XT (gfx1100)

**Verified Models:**
- Whisper (all sizes) via faster-whisper
- BERT, DistilBERT
- GPT-2, Llama, Mistral
- T5, BART

---

## About CTranslate2

CTranslate2 is a C++ and Python library for efficient inference with Transformer models. It implements a custom runtime that applies many performance optimization techniques:

**Core Features:**
- Weights quantization: FP16, BF16, INT16, INT8, AWQ (INT4)
- Layer fusion and padding removal
- Batch reordering and in-place operations
- Dynamic memory usage with caching allocators
- Parallel and asynchronous execution

**Supported Model Types:**
- **Encoder-decoder:** Transformer, M2M-100, NLLB, BART, mBART, Pegasus, T5, Whisper
- **Decoder-only:** GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, Llama, Mistral, Gemma, CodeGen, Falcon, Qwen2
- **Encoder-only:** BERT, DistilBERT, XLM-RoBERTa

For complete CTranslate2 documentation, see [opennmt.net/CTranslate2](https://opennmt.net/CTranslate2/).

---

## License

This project maintains the same MIT License as the upstream [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2).
