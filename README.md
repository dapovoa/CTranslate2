[![CI](https://github.com/OpenNMT/CTranslate2/workflows/CI/badge.svg)](https://github.com/OpenNMT/CTranslate2/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/ctranslate2.svg)](https://badge.fury.io/py/ctranslate2) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/CTranslate2/) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Forum](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.opennmt.net%2F)](https://forum.opennmt.net/)

# CTranslate2

## ROCm 7.x Support

This fork includes compatibility fixes for ROCm 7.0+ based on the official [ROCm/CTranslate2](https://github.com/ROCm/CTranslate2) repository. The original AMD fork targets ROCm 6.1, but significant API changes in ROCm 7.0+ required additional modifications.

### Changes for ROCm 7.0+

The hipBLAS API changed between ROCm 6.x and 7.x, requiring the following fixes:

1. **Type compatibility**: Changed `hipblasDatatype_t` to `hipDataType` in HIP macro definitions
2. **Compute type mappings**: Added `CUDA_COMPUTE_16F`, `CUDA_COMPUTE_32F`, `CUDA_COMPUTE_32I` for `hipblasComputeType_t`
3. **GEMM calls**: Updated all `hipblasGemmEx` and `hipblasGemmStridedBatchedEx` calls to use compute types instead of data types

### Modified Files

- `src/cuda2hip_macros.hpp`: Type definitions and compute type mappings
- `src/cuda/primitives.cu`: GEMM function calls updated for ROCm 7.x API

### Tested Configuration

- ROCm Version: 7.0.1
- GPU: AMD Radeon RX 7900 XT (gfx1100)
- Python: 3.12
- Build Type: Release with HIP support

### Building from Source

```bash
git clone --branch amd_dev --recursive https://github.com/dapovoa/CTranslate2.git
cd CTranslate2
mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH="/opt/rocm" \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DWITH_MKL=OFF \
      -DWITH_DNNL=OFF \
      -DWITH_OPENBLAS=OFF \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_HIP_ARCHITECTURES="gfx1100" \
      -DGPU_TARGETS="gfx1100" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=HIP \
      -DENABLE_CPU_DISPATCH=OFF \
      -DCMAKE_CXX_FLAGS="-O3" ..

make -j$(nproc) install

# Build Python wheel
cd ../python
pip install pybind11 wheel
python setup.py bdist_wheel
```

Replace `gfx1100` with your GPU architecture (`gfx90a`, `gfx942`, `gfx1030`, etc.).

### Pre-built Wheels

Pre-compiled Python wheels for ROCm 7.0.1 are available in the [Releases](https://github.com/dapovoa/CTranslate2/releases) section.

---

CTranslate2 is a C++ and Python library for efficient inference with Transformer models.

The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to [accelerate and reduce the memory usage](#benchmarks) of Transformer models on CPU and GPU.

The following model types are currently supported:

* Encoder-decoder models: Transformer base/big, M2M-100, NLLB, BART, mBART, Pegasus, T5, Whisper
* Decoder-only models: GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, Llama, Mistral, CodeGen, GPTBigCode, Falcon
* Encoder-only models: BERT, DistilBERT, XLM-RoBERTa

Compatible models should be first converted into an optimized model format. The library includes converters for multiple frameworks:

* [OpenNMT-py](https://opennmt.net/CTranslate2/guides/opennmt_py.html)
* [OpenNMT-tf](https://opennmt.net/CTranslate2/guides/opennmt_tf.html)
* [Fairseq](https://opennmt.net/CTranslate2/guides/fairseq.html)
* [Marian](https://opennmt.net/CTranslate2/guides/marian.html)
* [OPUS-MT](https://opennmt.net/CTranslate2/guides/opus_mt.html)
* [Transformers](https://opennmt.net/CTranslate2/guides/transformers.html)

The project is production-oriented and comes with [backward compatibility guarantees](https://opennmt.net/CTranslate2/versioning.html), but it also includes experimental features related to model compression and inference acceleration.

## Key features

* **Fast and efficient execution on CPU and GPU**<br/>The execution [is significantly faster and requires less resources](#benchmarks) than general-purpose deep learning frameworks on supported models and tasks thanks to many advanced optimizations: layer fusion, padding removal, batch reordering, in-place operations, caching mechanism, etc.
* **Quantization and reduced precision**<br/>The model serialization and computation support weights with [reduced precision](https://opennmt.net/CTranslate2/quantization.html): 16-bit floating points (FP16), 16-bit brain floating points (BF16), 16-bit integers (INT16), and 8-bit integers (INT8).
* **Multiple CPU architectures support**<br/>The project supports x86-64 and AArch64/ARM64 processors and integrates multiple backends that are optimized for these platforms: [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), [oneDNN](https://github.com/oneapi-src/oneDNN), [OpenBLAS](https://www.openblas.net/), [Ruy](https://github.com/google/ruy), and [Apple Accelerate](https://developer.apple.com/documentation/accelerate).
* **Automatic CPU detection and code dispatch**<br/>One binary can include multiple backends (e.g. Intel MKL and oneDNN) and instruction set architectures (e.g. AVX, AVX2) that are automatically selected at runtime based on the CPU information.
* **Parallel and asynchronous execution**<br/>Multiple batches can be processed in parallel and asynchronously using multiple GPUs or CPU cores.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Lightweight on disk**<br/>Quantization can make the models 4 times smaller on disk with minimal accuracy loss.
* **Simple integration**<br/>The project has few dependencies and exposes simple APIs in [Python](https://opennmt.net/CTranslate2/python/overview.html) and C++ to cover most integration needs.
* **Configurable and interactive decoding**<br/>[Advanced decoding features](https://opennmt.net/CTranslate2/decoding.html) allow autocompleting a partial sequence and returning alternatives at a specific location in the sequence.

Some of these features are difficult to achieve with standard deep learning frameworks and are the motivation for this project.

## Installation and usage

CTranslate2 can be installed with pip:

```bash
pip install ctranslate2
```

To build docker image for ROCm AMD GPUs

```bash
please checkout amd_dev branch;   cd docker_rocm;   docker build -t   rocm_ct2_v3.23.0 -f Dockerfile.rocm . 
```

The Python module is used to convert models and can translate or generate text with few lines of code:

```python
translator = ctranslate2.Translator(translation_model_path)
translator.translate_batch(tokens)

generator = ctranslate2.Generator(generation_model_path)
generator.generate_batch(start_tokens)
```

See the [documentation](https://opennmt.net/CTranslate2) for more information and examples.

## Benchmarks

We translate the En->De test set *newstest2014* with multiple models:

* [OpenNMT-tf WMT14](https://opennmt.net/Models-tf/#translation): a base Transformer trained with OpenNMT-tf on the WMT14 dataset (4.5M lines)
* [OpenNMT-py WMT14](https://opennmt.net/Models-py/#translation): a base Transformer trained with OpenNMT-py on the WMT14 dataset (4.5M lines)
* [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/en-de#opus-2020-02-26zip): a base Transformer trained with Marian on all OPUS data available on 2020-02-26 (81.9M lines)

The benchmark reports the number of target tokens generated per second (higher is better). The results are aggregated over multiple runs. See the [benchmark scripts](tools/benchmark) for more details and reproduce these numbers.

**Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.**

#### CPU

| | Tokens per second | Max. memory | BLEU |
| --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 209.2 | 2653MB | 26.93 |
| **OpenNMT-py WMT14 model** | | | |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1) | 275.8 | 2012MB | 26.77 |
| - int8 | 323.3 | 1359MB | 26.72 |
| CTranslate2 3.6.0 | 658.8 | 849MB | 26.77 |
| - int16 | 733.0 | 672MB | 26.82 |
| - int8 | 860.2 | 529MB | 26.78 |
| - int8 + vmap | 1126.2 | 598MB | 26.64 |
| **OPUS-MT model** | | | |
| Transformers 4.26.1 (with PyTorch 1.13.1) | 147.3 | 2332MB | 27.90 |
| Marian 1.11.0 | 344.5 | 7605MB | 27.93 |
| - int16 | 330.2 | 5901MB | 27.65 |
| - int8 | 355.8 | 4763MB | 27.27 |
| CTranslate2 3.6.0 | 525.0 | 721MB | 27.92 |
| - int16 | 596.1 | 660MB | 27.53 |
| - int8 | 696.1 | 516MB | 27.65 |

Executed with 4 threads on a [*c5.2xlarge*](https://aws.amazon.com/ec2/instance-types/c5/) Amazon EC2 instance equipped with an Intel(R) Xeon(R) Platinum 8275CL CPU.

#### GPU

| | Tokens per second | Max. GPU memory | Max. CPU memory | BLEU |
| --- | --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | | |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 1483.5 | 3031MB | 3122MB | 26.94 |
| **OpenNMT-py WMT14 model** | | | | |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1) | 1795.2 | 2973MB | 3099MB | 26.77 |
| FasterTransformer 5.3 | 6979.0 | 2402MB | 1131MB | 26.77 |
| - float16 | 8592.5 | 1360MB | 1135MB | 26.80 |
| CTranslate2 3.6.0 | 6634.7 | 1261MB | 953MB | 26.77 |
| - int8 | 8567.2 | 1005MB | 807MB | 26.85 |
| - float16 | 10990.7 | 941MB | 807MB | 26.77 |
| - int8 + float16 | 8725.4 | 813MB | 800MB | 26.83 |
| **OPUS-MT model** | | | | |
| Transformers 4.26.1 (with PyTorch 1.13.1) | 1022.9 | 4097MB | 2109MB | 27.90 |
| Marian 1.11.0 | 3241.0 | 3381MB | 2156MB | 27.92 |
| - float16 | 3962.4 | 3239MB | 1976MB | 27.94 |
| CTranslate2 3.6.0 | 5876.4 | 1197MB | 754MB | 27.92 |
| - int8 | 7521.9 | 1005MB | 792MB | 27.79 |
| - float16 | 9296.7 | 909MB | 814MB | 27.90 |
| - int8 + float16 | 8362.7 | 813MB | 766MB | 27.90 |

Executed with CUDA 11 on a [*g5.xlarge*](https://aws.amazon.com/ec2/instance-types/g5/) Amazon EC2 instance equipped with a NVIDIA A10G GPU (driver version: 510.47.03).

## Additional resources

* [Documentation](https://opennmt.net/CTranslate2)
* [Forum](https://forum.opennmt.net)
* [Gitter](https://gitter.im/OpenNMT/CTranslate2)
