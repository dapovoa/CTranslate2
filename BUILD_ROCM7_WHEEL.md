# BUILD ROCm7 Wheel (Essencial)

```bash
cd /home/dapovoa/github/CTranslate2
git checkout amd_dev_v4.6.0_rocm7
rm -rf build-rocm7-wheel python/build python/dist

cmake -S . -B build-rocm7-wheel \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DWITH_DNNL=OFF \
  -DWITH_OPENBLAS=OFF \
  -DOPENMP_RUNTIME=NONE \
  -DENABLE_CPU_DISPATCH=OFF \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DGPU_TARGETS=gfx1100 \
  -DBUILD_CLI=OFF

cmake --build build-rocm7-wheel -j"$(nproc)"

python3.12 -m venv .venv-build
source .venv-build/bin/activate
pip install -U pip setuptools wheel pybind11

cd python
CT2_LOCAL_VERSION=rocm7.0 \
CTRANSLATE2_ROOT=/home/dapovoa/github/CTranslate2/build-rocm7-wheel \
python setup.py bdist_wheel

ls -lh dist/*.whl
```

Saida esperada:

```text
dist/ctranslate2-4.6.0+rocm7.0-cp312-cp312-linux_x86_64.whl
```
