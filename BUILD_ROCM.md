# Building CTranslate2 v4.6.0 for AMD ROCm

This guide explains how to build CTranslate2 v4.6.0 with full AMD ROCm support, including MIOpen for GPU-accelerated operations (Conv1D, Flash Attention, etc.).

## Prerequisites

### System Requirements
- AMD GPU with ROCm support (tested on RX 7900 XTX - gfx1100)
- Ubuntu 22.04/24.04 LTS (or compatible Linux distribution)
- ROCm 6.3.0 or later
- Python 3.8+ (tested with Python 3.12)

### Install ROCm Dependencies

```bash
# Install ROCm core packages
sudo apt-get update
sudo apt-get install -y \
    rocm-dev \
    rocm-libs \
    miopen-hip \
    miopen-hip-dev \
    hipblas-dev \
    rocblas-dev

# Verify ROCm installation
rocm-smi --showdriverversion
```

## Build Instructions

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/dapovoa/CTranslate2.git
cd CTranslate2

# Create build directory
mkdir -p build
cd build
```

### 2. CMake Configuration

**Important:** You must enable `WITH_CUDNN=ON` to get MIOpen support for Conv1D operations.

```bash
cmake -DCMAKE_PREFIX_PATH="/opt/rocm" \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DWITH_MKL=OFF \
      -DWITH_DNNL=OFF \
      -DWITH_OPENBLAS=OFF \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_HIP_ARCHITECTURES="gfx1100" \
      -DAMDGPU_TARGETS="gfx1100" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=HIP \
      -DENABLE_CPU_DISPATCH=OFF \
      -DCMAKE_CXX_FLAGS="-O3" ..
```

**Configuration flags explained:**
- `WITH_CUDA=ON` - Enable GPU support (uses ROCm/HIP)
- `WITH_CUDNN=ON` - Enable MIOpen (AMD's cuDNN equivalent) for Conv1D operations
- `WITH_MKL=OFF` - Disable Intel MKL (not needed for AMD)
- `DOPENMP_RUNTIME=COMP` - Use LLVM/ROCm OpenMP instead of Intel OpenMP
- `CMAKE_PREFIX_PATH="/opt/rocm"` - Help CMake find ROCm libraries
- `AMDGPU_TARGETS=gfx1100` - Compile for your GPU architecture (adjust as needed)

**Common GPU architectures:**
- RX 7900 XT/XTX: `gfx1100`
- RX 6900 XT: `gfx1030`
- RX 6800/6800 XT: `gfx1030`
- Check yours with: `rocminfo | grep gfx`

### 3. Compile

```bash
make -j$(nproc)
```

### 4. Install System-wide (Required for Python binding)

```bash
sudo make install
sudo ldconfig
```

This installs `libctranslate2.so` to `/usr/local/lib/` with MIOpen support.

## Building Python Package

### 1. Install Python Dependencies

```bash
cd ../python
pip install pybind11 setuptools wheel
```

### 2. Build Wheel

The `setup.py` has been modified to automatically include MIOpen and hipblas libraries for Linux builds.

```bash
export CTRANSLATE2_ROOT=/path/to/CTranslate2/build
python setup.py bdist_wheel
```

### 3. Install

```bash
pip install dist/ctranslate2-*.whl
```

## Verification

Test that MIOpen is properly linked:

```python
import ctranslate2._ext
import subprocess

so_path = ctranslate2._ext.__file__
result = subprocess.run(['ldd', so_path], capture_output=True, text=True)

# Should show libMIOpen.so.1
for line in result.stdout.split('\n'):
    if 'MIOpen' in line:
        print(line)
```

Test Conv1D operations on GPU:

```python
import ctranslate2
import numpy as np

# Check supported compute types
devices = ctranslate2.get_supported_compute_types('cuda')
print(f'Supported compute types: {devices}')

# Load a Whisper model and test transcription
# (requires downloaded Whisper model in CTranslate2 format)
```

## Troubleshooting

### Error: "Conv1D on GPU currently requires the cuDNN library"

This means `WITH_CUDNN=ON` was not enabled during CMake configuration, or MIOpen was not found.

**Solution:**
1. Verify MIOpen is installed: `dpkg -l | grep miopen`
2. Reconfigure with `WITH_CUDNN=ON`
3. Check CMake output for: `WITH_CUDNN:BOOL=ON`
4. Rebuild completely

### Error: "hipblas not found" during CMake

**Solution:**
```bash
sudo apt-get install hipblas-dev
```

### Python package doesn't link MIOpen

**Solution:**
1. Ensure `sudo make install` was run for libctranslate2
2. Verify `/usr/local/lib/libctranslate2.so.3` has MIOpen: `ldd /usr/local/lib/libctranslate2.so.3 | grep MIOpen`
3. Rebuild Python package with clean build: `rm -rf build dist && python setup.py bdist_wheel`

## What's New in v4.6.0 ROCm Build

### Features
- Full compatibility with upstream CTranslate2 v4.6.0
- Flash Attention v2 support (GPU-accelerated)
- Whisper large-v3-turbo support
- wav2vec2bert model support
- NCCL operations for multi-GPU (experimental)
- Improved bfloat16 support

### Known Limitations
- AWQ quantization (INT4) is **disabled** - uses NVIDIA PTX assembly incompatible with ROCm
  - Models with AWQ will throw runtime error: `"AWQ quantization is not supported in this ROCm build"`
  - Alternative: Use CT2 native INT8 quantization instead

### Recommended Settings
- **Compute type**: `int8_float16` or `float16` for best performance
- **VRAM**: 12GB+ for Whisper large models
- **ROCm version**: 6.3.0+ for best stability

## Performance Notes

- **int8_float16** compute type recommended for best performance on AMD GPUs
- Whisper large-v3 and large-v3-turbo run efficiently with 12GB+ VRAM
- Flash Attention v2 provides significant speedup for long sequences
- ROCm 6.3+ recommended for best stability

## Credits

- Original CTranslate2: [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2)
- ROCm v4.6.0 port and compatibility fixes: [@dapovoa](https://github.com/dapovoa)
