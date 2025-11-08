[![CI](https://github.com/dapovoa/CTranslate2/workflows/CI/badge.svg)](https://github.com/dapovoa/CTranslate2/actions?query=workflow%3ACI) [![Release](https://img.shields.io/github/v/release/dapovoa/CTranslate2)](https://github.com/dapovoa/CTranslate2/releases/latest) [![ROCm](https://img.shields.io/badge/ROCm-6.3.1-red.svg)](BUILD_ROCM.md) [![CTranslate2](https://img.shields.io/badge/CTranslate2-v4.6.0-blue.svg)](https://github.com/OpenNMT/CTranslate2) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# CTranslate2 - AMD ROCm Fork

This is a fork of [ROCm/CTranslate2](https://github.com/ROCm/CTranslate2), which is AMD's official ROCm port of [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2). It brings CTranslate2 v4.6.0 support, modern ROCm compatibility, and proper MIOpen integration for Whisper models.

**Latest Release:** [v4.6.0-rocm6.3.1](https://github.com/dapovoa/CTranslate2/releases/latest) | **Upstream:** [ROCm/CTranslate2](https://github.com/ROCm/CTranslate2)

---

## What This Adds

The official ROCm fork is at CTranslate2 v3.23.0 and targets ROCm 6.1. This fork updates things significantly:

**Core Upgrades**
- **CTranslate2 v4.6.0** - Jumped from v3.23.0, bringing 70+ upstream commits
- **Modern ROCm** - Works with ROCm 6.2.4, 6.3.x, and 7.0.1
- **MIOpen working** - Fixed the "Conv1D requires cuDNN" error that breaks Whisper
- **GPU-only builds** - Removed CPU dispatch kernels for cleaner implementation
- **Automated building** - The `build_rocm.sh` script handles everything

**New Stuff from v4.6.0**
- Support for newer models: Gemma 2, Qwen2, Phi-3, Mistral Nemo
- Flash Attention for faster inference
- AWQ quantization (INT4) for smaller models on disk
- Tensor parallelism if you have multiple GPUs

Full details in the [release notes](https://github.com/dapovoa/CTranslate2/releases/latest).

---

## Quick Start

### Using Pre-Built Wheels

Grab the wheel from [Releases](https://github.com/dapovoa/CTranslate2/releases):

```bash
pip install ctranslate2-4.6.0+rocm6.3.1-cp312-cp312-linux_x86_64.whl
```

**What you need:**
- ROCm 6.3.1 (also works with 6.2.4 and 7.0.1)
- AMD Radeon RX 7900 XTX or similar RDNA3 GPU (gfx1100)
- Python 3.12
- Ubuntu 24.04 LTS or compatible

### Building from Source

```bash
git clone --branch amd_dev_v4.6.0_rocm6.3 --recursive https://github.com/dapovoa/CTranslate2.git
cd CTranslate2
./build_rocm.sh
```

The script detects your GPU and builds everything. Python wheel ends up in `python/dist/`.

For manual build steps, check [BUILD_ROCM.md](BUILD_ROCM.md).

---

## Technical Details

**ROCm 6.3+ Compatibility**

ROCm 6.3 changed a bunch of header locations and API signatures. This fork handles that with automatic hipBLAS version detection at compile time and updated include paths. It's been tested on ROCm 6.2.4, 6.3.0, 6.3.1, and 7.0.1.

**CUDA to HIP Translation**

Since CTranslate2 was originally CUDA-only, there are custom compatibility headers that map CUDA calls to HIP:
- `cuda2hip_macros.hpp` for macro translations
- `cuda2hip_types.hpp` for type compatibility
- `cuda2hip_device.hpp` for device function mappings

Plus namespace remapping for CUB to hipCUB and cuRAND to hipRAND.

**MIOpen Integration**

The upstream ROCm fork had incomplete MIOpen linking, which caused Conv1D operations to fail. This is now properly configured in CMakeLists.txt, so Whisper models work correctly. Tested extensively with `faster-whisper` for real-time audio transcription.

---

## What's Been Tested

**Main setup:**
- ROCm 6.3.1
- AMD Radeon RX 7900 XTX (gfx1100)
- Ubuntu 24.04 LTS
- Python 3.12

**Also verified on:**
- ROCm 6.2.4 with RX 7900 XT
- ROCm 7.0.1 with RX 7900 XT

**Models that work:**
- Whisper (all sizes) through faster-whisper
- BERT and DistilBERT
- GPT-2, Llama, Mistral
- T5 and BART

---

## About CTranslate2

CTranslate2 is a library focused on fast inference for Transformer models. It does a lot of optimization work under the hood: weights quantization (FP16, BF16, INT8, INT4), layer fusion, smart batching, and efficient memory management.

It supports pretty much every common Transformer architecture: encoder-decoder models (Transformer, BART, T5, Whisper), decoder-only models (GPT-2, Llama, Mistral, Gemma, Qwen2), and encoder-only models (BERT).

Full docs at [opennmt.net/CTranslate2](https://opennmt.net/CTranslate2/).

---

## License

MIT License, same as the upstream OpenNMT project.
