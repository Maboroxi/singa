# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Behavioral checks for the optional local ONNX storage feature."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from singa.storage import Neurstore

try:
    from singa import neurstore_wrap
    from singa._storage_config import enabled
    AVAILABLE = enabled
except ImportError:
    AVAILABLE = False


def fixture():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])
    w = numpy_helper.from_array(np.linspace(-0.4, 0.7, 12, dtype=np.float32).reshape(4, 3), 'w')
    b = numpy_helper.from_array(np.array([0.1, -0.2, 0.3], dtype=np.float32), 'b')
    graph = helper.make_graph([
        helper.make_node('MatMul', ['x', 'w'], ['a']),
        helper.make_node('Add', ['a', 'b'], ['z']),
        helper.make_node('Relu', ['z'], ['y']),
    ], 'storage_fixture', [x], [y], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 6
    return model


@unittest.skipUnless(AVAILABLE, 'USE_ONNX_STORAGE is disabled')
class TestNeurstore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'store'
        self.store = Neurstore(self.path)
        self.model = fixture()

    def tearDown(self):
        self.temp.cleanup()

    def check_model(self, actual):
        onnx.checker.check_model(actual)
        weights = {w.name: w for w in actual.graph.initializer}
        for expected in self.model.graph.initializer:
            weight = weights[expected.name]
            self.assertEqual(weight.data_type, expected.data_type)
            self.assertEqual(list(weight.dims), list(expected.dims))
            a, b = numpy_helper.to_array(expected), numpy_helper.to_array(weight)
            if a.dtype == np.float32 and a.size and np.isfinite(a).all():
                np.testing.assert_allclose(a, b, rtol=0, atol=1.2e-6)
            else:
                self.assertEqual(weight.SerializeToString(), expected.SerializeToString())
        a, b = onnx.ModelProto(), onnx.ModelProto()
        a.CopyFrom(self.model)
        b.CopyFrom(actual)
        a.graph.ClearField('initializer')
        b.graph.ClearField('initializer')
        self.assertEqual(a, b)

    def test_roundtrip_and_input_unchanged(self):
        before = self.model.SerializeToString()
        self.assertIsNone(self.store.saveModel('model', self.model, tolerance=1e-6))
        self.assertEqual(self.model.SerializeToString(), before)
        self.check_model(self.store.loadModel('model'))

    def test_file_path_and_fresh_process(self):
        source = self.root / 'input.onnx'
        onnx.save(self.model, str(source))
        self.store.saveModel('model', source, tolerance=1e-6)
        output = self.root / 'restored.onnx'
        subprocess.run([sys.executable, '-c',
            'import sys,onnx; from singa.storage import Neurstore; '
            'onnx.save(Neurstore(sys.argv[1]).loadModel("model"),sys.argv[2])',
            str(self.path), str(output)], check=True, timeout=60)
        self.check_model(onnx.load(str(output)))

    def test_shared_bases_and_reopen_before_second_save(self):
        self.store.saveModel('first', self.model, 1e-6)
        before = {p.name: struct.unpack_from('=q', p.read_bytes(), 40)[0]
                  for p in self.path.glob('*.index')}
        Neurstore(self.path).saveModel('second', self.model, 1e-6)
        after = {p.name: struct.unpack_from('=q', p.read_bytes(), 40)[0]
                 for p in self.path.glob('*.index')}
        self.assertEqual(before, after)
        self.assertTrue(list((self.path / 'router').glob('*.router')))
        self.check_model(Neurstore(self.path).loadModel('first'))
        self.check_model(Neurstore(self.path).loadModel('second'))

    def test_constant_zero_scalar_and_mixed_types(self):
        values = [np.zeros(7, np.float32), np.full(5, 0.125, np.float32),
                  np.array(0.3, np.float32), np.ones(2, np.float16),
                  np.array([1.25], np.float64), np.array([2**50], np.int64),
                  np.array([True, False]), np.empty((0, 3), np.float32),
                  np.array([np.nan, np.inf], np.float32)]
        for i, value in enumerate(values):
            self.model.graph.initializer.extend([numpy_helper.from_array(value, 'extra%d' % i)])
        self.store.saveModel('mixed', self.model, 1e-6)
        self.check_model(self.store.loadModel('mixed'))

    def test_embedded_float_data(self):
        weight = self.model.graph.initializer[0]
        values = numpy_helper.to_array(weight).reshape(-1).tolist()
        weight.ClearField('raw_data')
        weight.float_data.extend(values)
        self.store.saveModel('typed', self.model, 1e-6)
        self.check_model(self.store.loadModel('typed'))

    def test_graph_without_compressed_weights(self):
        for weight in self.model.graph.initializer:
            weight.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(weight).astype(np.float16), weight.name))
        # Storage validates ONNX structure, not inference type compatibility.
        self.store.saveModel('retained', self.model, 1e-6)
        self.check_model(self.store.loadModel('retained'))

    def test_singa_inference(self):
        from singa import sonnx
        x = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4)
        before = sonnx.prepare(self.model, device='CPU').run([x])[0]
        self.store.saveModel('infer', self.model, 1e-6)
        restored = self.store.loadModel('infer')
        after = sonnx.prepare(restored, device='CPU').run([x])[0]
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)

    def test_duplicate_name_and_existing_model_survives(self):
        self.store.saveModel('model', self.model, 1e-6)
        with self.assertRaises(ValueError):
            self.store.saveModel('model', self.model, 1e-6)
        self.check_model(self.store.loadModel('model'))

    def test_bad_names_and_tolerances(self):
        for name in ('', '..', '../x', 'a/b', 'a\\b', 'a\0b'):
            with self.assertRaises(ValueError):
                self.store.saveModel(name, self.model)
        for tolerance in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.store.saveModel('bad', self.model, tolerance)
        self.assertFalse(list(self.path.glob('*.structure')))

    def test_missing_model(self):
        with self.assertRaises(RuntimeError):
            self.store.loadModel('missing')

    def test_external_weights_rejected(self):
        weight = self.model.graph.initializer[0]
        weight.data_location = TensorProto.EXTERNAL
        entry = weight.external_data.add()
        entry.key, entry.value = 'location', 'weights.bin'
        with self.assertRaises(ValueError):
            self.store.saveModel('external', self.model)

    def test_missing_index_does_not_create_empty_index(self):
        self.store.saveModel('model', self.model, 1e-6)
        index = next(self.path.glob('*.index'))
        index.unlink()
        with self.assertRaises(RuntimeError):
            self.store.loadModel('model')
        self.assertFalse(index.exists())

    def test_truncated_files_raise(self):
        self.store.saveModel('model', self.model, 1e-6)
        paths = [self.path / 'model.structure', self.path / 'model.tensors',
                 next(self.path.glob('*.index')), next((self.path / 'router').glob('*.router'))]
        for path in paths:
            original = path.read_bytes()
            try:
                path.write_bytes(original[:3])
                with self.assertRaises(RuntimeError, msg=str(path)):
                    Neurstore(self.path).loadModel('model')
            finally:
                path.write_bytes(original)

    def _check_index_error_in_subprocess(self, index, original):
        # A native crash must fail this test without terminating the test runner.
        program = """
import resource
import sys
from pathlib import Path
from singa.storage import Neurstore
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
original = sys.stdin.buffer.read()
store = Neurstore(sys.argv[1])
try:
    store.loadModel('model')
except RuntimeError:
    print('caught RuntimeError', flush=True)
else:
    raise AssertionError('Invalid index was accepted')
Path(sys.argv[2]).write_bytes(original)
assert store.loadModel('model').graph.initializer
print('valid index reloaded', flush=True)
"""
        result = subprocess.run(
            [sys.executable, '-c', program, str(self.path), str(index)],
            input=original, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30)
        self.assertEqual(result.returncode, 0,
                         result.stderr.decode(errors='replace'))
        self.assertIn(b'caught RuntimeError', result.stdout)
        self.assertIn(b'valid index reloaded', result.stdout)

    def test_truncated_index_cleanup_in_subprocess(self):
        self.store.saveModel('model', self.model, 1e-6)
        index = next(self.path.glob('*.index'))
        original = index.read_bytes()
        size_t = struct.calcsize('P')
        # The legacy wrapper has four ints, two int64s and two size_t fields.
        wrapper_size = 4 * 4 + 2 * 8 + 2 * size_t
        count_end = wrapper_size + 3 * size_t
        # HNSW header: ten size_t fields, two 32-bit fields and one double.
        header_end = wrapper_size + 10 * size_t + 2 * 4 + 8
        cut_points = [count_end, count_end - 1, count_end + 1,
                      header_end - 1, header_end, header_end + 1,
                      len(original) - 1, 3]
        try:
            for cut in cut_points:
                with self.subTest(cut=cut):
                    index.write_bytes(original[:cut])
                    self._check_index_error_in_subprocess(index, original)
        finally:
            index.write_bytes(original)
        self.check_model(self.store.loadModel('model'))

    def test_invalid_index_header_cleanup_in_subprocess(self):
        self.store.saveModel('model', self.model, 1e-6)
        index = next(self.path.glob('*.index'))
        original = index.read_bytes()
        wrapper_size = 4 * 4 + 2 * 8 + 2 * struct.calcsize('P')
        corrupted = bytearray(original)
        corrupted[wrapper_size] = 1  # offsetLevel0 must be zero.
        try:
            index.write_bytes(corrupted)
            self._check_index_error_in_subprocess(index, original)
        finally:
            index.write_bytes(original)
        self.check_model(self.store.loadModel('model'))

    def test_write_failure_keeps_existing_models(self):
        self.store.saveModel('model', self.model, 1e-6)
        self.path.chmod(0o555)
        try:
            with self.assertRaises(RuntimeError):
                self.store.saveModel('unwritable', self.model, 1e-6)
        finally:
            self.path.chmod(0o755)
        self.assertFalse((self.path / 'unwritable.structure').exists())
        self.check_model(self.store.loadModel('model'))

    def test_nested_external_tensor_rejected(self):
        node = self.model.graph.node.add()
        node.op_type = 'Constant'
        node.output.append('unused_constant')
        attribute = node.attribute.add()
        attribute.name = 'value'
        attribute.type = onnx.AttributeProto.TENSOR
        attribute.t.name = 'external_constant'
        attribute.t.data_location = TensorProto.EXTERNAL
        with self.assertRaises(ValueError):
            self.store.saveModel('nested', self.model)



class TestStorageDisabled(unittest.TestCase):
    def test_clear_error_when_native_extension_unavailable(self):
        # Exercise the real facade against a package with no native extension,
        # even when this test process itself has storage enabled.
        import importlib.util
        import types
        import singa.storage
        package = types.ModuleType('singa_storage_disabled_test')
        package.__path__ = []
        name = package.__name__ + '.storage'
        spec = importlib.util.spec_from_file_location(name, singa.storage.__file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[package.__name__] = package
        try:
            spec.loader.exec_module(module)
            with self.assertRaisesRegex(ImportError, 'USE_ONNX_STORAGE=ON'):
                module.Neurstore('/tmp/unused-store')
        finally:
            sys.modules.pop(package.__name__, None)


if __name__ == '__main__':
    unittest.main()
