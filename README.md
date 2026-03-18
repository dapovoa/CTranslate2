[![Release](https://img.shields.io/github/v/release/dapovoa/CTranslate2)](https://github.com/dapovoa/CTranslate2/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# CTranslate2 ROCm Fork

ROCm-focused fork of CTranslate2, based on upstream OpenNMT and ROCm work, with practical fixes for modern AMD GPU environments.

## Upstream References

- Original project: [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2)
- ROCm upstream fork: [ROCm/CTranslate2](https://github.com/ROCm/CTranslate2)

## Scope of This Fork

- Base line: CTranslate2 v4.6.0
- Target: AMD ROCm environments on Linux
- Focus: stable build/runtime path for Whisper and general inference workloads
- Packaging: wheel builds for local deployment and release distribution

## Documentation

- ROCm build notes: [BUILD_ROCM.md](BUILD_ROCM.md)
- ROCm 7 wheel build flow: [BUILD_ROCM7_WHEEL.md](BUILD_ROCM7_WHEEL.md)

## Quick Start

Install from a release wheel:

```bash
pip install ctranslate2-<version>-cp312-cp312-linux_x86_64.whl
```

Build from source:

```bash
git clone --recursive https://github.com/dapovoa/CTranslate2.git
cd CTranslate2
./build_rocm.sh
```

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
