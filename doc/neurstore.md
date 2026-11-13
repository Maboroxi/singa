# Local ONNX model storage

`singa.storage.Neurstore` stores ONNX graphs and weights in a local directory,
using NeurStore's shared UINT8 bases and quantized residuals. It accepts an
ONNX `ModelProto` or a file path and restores an ONNX `ModelProto`.

## Build

This feature defaults to **OFF**. Add the following options to an otherwise
working SINGA CMake configuration (the paths below are for `singa-py39-dev`):

```bash
cmake -S /singa -B /singa/build/singa-py39 \
  -DUSE_ONNX_STORAGE=ON \
  -DNEURSTORE_EIGEN_INCLUDE_DIR=/pg_neurstore/external/eigen
cmake --build /singa/build/singa-py39 --target _singa_wrap _neurstore_wrap -j1
export PYTHONPATH=/singa/build/singa-py39/python
```

Retain the existing Python, Protobuf, BLAS, SWIG and CPU options. The validated
environment uses Python 3.9.25, ONNX 1.6.0, NumPy 1.23.5, Protobuf 3.9.2 and SWIG
3.0.12. The native storage library needs Eigen, OpenMP, a C++17 compiler and an
AVX2/FMA CPU. ONNX C++ sources are generated from the bundled schema with SINGA's
selected `protoc`; do not mix generated code or libraries from another Protobuf
version. PostgreSQL, ONNX Runtime, OpenCV and tokenizers are not storage dependencies.

The separate SWIG module is installed alongside SINGA's regular binding by the
existing Python package build. A disabled build can import `singa.storage`;
constructing `Neurstore` reports that `USE_ONNX_STORAGE=ON` is required.

## Use

```python
from singa.storage import Neurstore
from singa import sonnx

store = Neurstore('/tmp/model-store')
store.saveModel('classifier', '/tmp/classifier.onnx', tolerance=1e-6)
restored = store.loadModel('classifier')
runtime = sonnx.SONNXModel(restored)
runtime.eval()
```

`saveModel` returns `None` after writing the shared indexes and model files.
`loadModel` returns an ONNX model; use `onnx.save` to export it. SINGA execution
still depends on SINGA's ONNX operator and tensor-name support. To store a SINGA
computation, first use `sonnx.to_onnx(inputs, outputs)`.

Names must be nonempty basenames and cannot contain `/`, `\\` or NUL. Duplicate
names are rejected. Input ONNX objects are copied. Models with external-data
references, including tensors inside subgraphs/attributes, must be embedded
before saving. Invalid inputs raise `ValueError`/ONNX validation errors; native
storage errors raise `RuntimeError` and Python path I/O errors retain their
normal exception types.

Only finite, nonempty FLOAT32 top-level initializers are compressed. Inline
`float_data` is normalized. Other weights, including FLOAT16, DOUBLE, integers,
empty tensors and nonfinite values, remain in the structure unchanged. Restored
initializer order may differ; names, shapes, dtypes and graph links are retained.
Training classes, optimizer state and training progress are not reconstructed;
use `Model.save_states/load_states` for existing checkpoint behavior.

## Persistence and precision

Each model has `<name>.structure` and `<name>.tensors`. Shared bases and HNSW
graphs live in `<element_count>_<index_id>.index`; routing state lives in
`router/<element_count>.router`. Preserve the entire directory when moving a
store. Complete legacy NeurStore stores are readable; lost legacy dtype metadata
cannot be recovered. New mixed-dtype stores do not promise compatibility with
the old reader, which clears retained initializers.

The first version processes tensors sequentially, retaining pending-base insertion
after the first pass. It supports a single writer per directory; separate
instances/processes must not write concurrently. Each operation reopens the
index manager, so a successful save can be recovered by a new process without
an explicit flush. Writes use same-directory temporary files and rename. Indexes
are flushed before the model pair, with `.structure` published last. Failed
saves may leave unreferenced bases; crashes can leave temporary files or an
incomplete pair. No multi-file transaction or power-loss guarantee is provided.

`tolerance` must be finite and positive. It controls the existing residual
quantizer and does **not** unconditionally bound weight error. Its bit-width
formula is `ceil(log2(R/(2*tolerance)))` clamped to 1..32, while the actual step
is `R/(2**bits-1)`. The known `R=4*tolerance` example can incur `2*tolerance`
error. Validate measured weight/output errors for the intended model. Do not
claim a storage reduction from the model pair alone: count indexes and routers.

## Checks

```bash
cd /singa/test/python
python -m unittest -v test_storage
```

Tests cover process-independent restoration, shared bases, typed FLOAT data,
mixed dtypes, SINGA CPU inference and error cases. They skip storage-dependent
cases in a disabled build; the missing-feature error test still runs.

Source provenance and vendored adaptations are recorded in
`third_party/neurstore/ORIGIN.md`.

## Validation record (2026-10-31)

Verified in `singa-py39-dev` using Python 3.9.25:

- Storage enabled: all 16 behavioral tests passed, including fresh-process
  recovery, SINGA execution, error cases and the unavailable-feature diagnostic.
- Storage disabled: `_singa_wrap` and `test_singa` built in the separate
  `build/singa-py39-storage-off` directory; all 171 native tests passed.
  The feature test passed and 15 storage-dependent tests were skipped.
- The CMake-generated Python package contained `_neurstore_wrap.so` and passed
  all 16 storage tests using its isolated package directory.
- A fresh process loaded a store produced by the original NeurStore binding.
  The fixture's maximum weight error was `9.5367431640625e-7`, and its maximum
  output error was `1.0132789611816406e-6`, with `tolerance=1e-6`.
- `SONNXModel` execution and all six import orders of `onnx`, `neurstore` and
  `singa.storage` passed actual restoration/inference checks. Dynamic dependency
  inspection showed no database, ORT, OpenCV or tokenizer dependency.

Logs are in `/pg_neurstore/build/singa-py39-results/storage-*.log` and
`storage-port/metrics.json`. These results cover the small CPU fixtures; they
are not a large-model, GPU, distributed or compression-ratio benchmark.
