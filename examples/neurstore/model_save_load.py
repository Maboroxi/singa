# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Save a tiny ONNX model locally and run the restored model with SINGA."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from singa import sonnx, tensor
from singa.storage import Neurstore


def make_model():
    """Build y = ReLU(x @ weight + bias), with two rows of four features."""
    weight = np.linspace(-0.4, 0.7, 12, dtype=np.float32).reshape(4, 3)
    bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['x', 'weight'], ['a']),
         helper.make_node('Add', ['a', 'bias'], ['z']),
         helper.make_node('Relu', ['z'], ['y'])],
        'neurstore_example',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(weight, 'weight'),
         numpy_helper.from_array(bias, 'bias')])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        help='Artifact directory; use a fresh directory per run.')
    args = parser.parse_args()
    output_dir = (args.output_dir if args.output_dir is not None else
                  Path(tempfile.mkdtemp(prefix='singa-neurstore-example-')))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original = make_model()
    original_path = output_dir / 'original.onnx'
    onnx.save(original, str(original_path))

    # saveModel also accepts original (the ModelProto) instead of this path.
    store_path = output_dir / 'store'
    store = Neurstore(store_path)
    store.saveModel('demo', original_path, tolerance=1e-6)

    # Reopen the persisted store and recover an ONNX ModelProto.
    restored = Neurstore(store_path).loadModel('demo')
    onnx.checker.check_model(restored)
    onnx.save(restored, str(output_dir / 'restored.onnx'))

    inputs = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4)
    expected = sonnx.prepare(original, device='CPU').run([inputs])[0]
    runtime = sonnx.SONNXModel(restored)
    runtime.eval()
    actual = tensor.to_numpy(runtime(tensor.from_numpy(inputs))[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    print('Restored model output:')
    print(actual)
    print('Maximum output error: %.8g' % np.max(np.abs(actual - expected)))
    print('Artifacts: %s' % output_dir)
    print('Save, load, ONNX validation and SINGA inference passed.')


if __name__ == '__main__':
    main()
