# Neurstore local model storage example

This CPU example creates a small ONNX `MatMul → Add → ReLU` model, saves it
through `Neurstore.saveModel`, reopens the local store, and restores it through
`loadModel`. It validates the ONNX graph and compares SINGA inference outputs
before and after storage. No model or dataset download is needed.

Build SINGA with `USE_ONNX_STORAGE=ON` and use its matching Python bindings;
see [the storage build instructions](../../doc/neurstore.md#build).
In the prepared `singa-py39-dev` container, run from `/singa`:

```bash
export PYTHONPATH=/singa/build/singa-py39/python
python examples/neurstore/model_save_load.py
```

Each run creates a new temporary artifact directory and prints its path.
To select a directory explicitly:

```bash
python examples/neurstore/model_save_load.py --output-dir /tmp/neurstore-demo
```

Use a fresh directory for each run because duplicate model names are rejected.
Artifacts include `original.onnx`, `restored.onnx`, and a `store/` directory
containing `demo.structure`, `demo.tensors`, shared `.index` files and `router/`.
Preserve the whole store directory when moving the stored model.

The example uses `tolerance=1e-6` and checks predictions with `rtol=1e-5` and
`atol=1e-5`. These are separate checks; the quantization tolerance is not an
unconditional weight-error guarantee. The script exits with an error if graph
validation or output comparison fails.
