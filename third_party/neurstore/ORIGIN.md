# NeurStore storage core provenance

Copied from the local NeurStore checkout, Git HEAD `7f16d7516764205a7d7f04f912a4f68661c20b81`, including its existing
uncommitted compressor changes. This is a source snapshot, not a pristine release.
The source checkout has no top-level license file; no additional license grant
is inferred. Keep upstream attribution and resolve distribution licensing before
an upstream Apache release. Custom HNSW files retain their bundled license.
The ONNX schema is from the checkout's external/onnx/onnx_protobuf/onnx/onnx.proto
and retains its original header.

Local adaptations: storage-only build; sequential tensor codec entry point;
checked facade deserialization and input validation; atomic index/router writes;
router and HNSW header checks; safe logarithm/UINT32 quantization arithmetic.
The residual codec's known tolerance limitation is retained.
