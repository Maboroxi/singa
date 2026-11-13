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

"""Local compressed ONNX storage, enabled with USE_ONNX_STORAGE.

This stores graphs and weights. Training checkpoints remain the responsibility
of Model.save_states/load_states. Stores support a single writer.
"""
import math
import os


class Neurstore:
    """Open a local store; saveModel/loadModel operate on named ONNX models."""

    def __init__(self, store_path):
        try:
            from ._storage_config import enabled
            if not enabled:
                raise ImportError('ONNX storage is disabled in this build')
            from . import neurstore_wrap
        except ImportError as exc:
            raise ImportError(
                "Neurstore requires SINGA built with USE_ONNX_STORAGE=ON "
                "and its native storage extension on the library path"
            ) from exc
        path = os.fspath(store_path)
        if not isinstance(path, str) or not path or '\x00' in path:
            raise ValueError('store_path must be a nonempty text path')
        self._native = neurstore_wrap.Neurstore(os.path.abspath(path))

    @staticmethod
    def _name(name):
        if not isinstance(name, str):
            raise TypeError('model_name must be a string')
        if not name or name in ('.', '..') or any(c in name for c in ('/', '\\', '\x00')):
            raise ValueError('model_name must be a nonempty basename')
        return name

    @staticmethod
    def _check_embedded(message):
        # Visit tensor attributes and nested graphs as well as initializers.
        if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
            if message.data_location == 1 or message.external_data:
                raise ValueError('External ONNX weights must be embedded before saving')
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                children = value if field.label == field.LABEL_REPEATED else (value,)
                for child in children:
                    Neurstore._check_embedded(child)

    def saveModel(self, model_name, model, tolerance=5.96e-8):
        """Save an ONNX ModelProto or file path, without modifying the input.

        Only finite nonempty FLOAT32 initializers are compressed. Other weights
        are preserved exactly. tolerance is a positive quantization parameter,
        not an unconditional maximum-error guarantee. Success returns None.
        Duplicate names and invalid inputs raise; native I/O errors also raise.
        """
        import numpy as np
        import onnx
        from onnx import numpy_helper

        name = self._name(model_name)
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite')
        copy = onnx.ModelProto()
        if isinstance(model, onnx.ModelProto):
            copy.CopyFrom(model)
        elif isinstance(model, (str, os.PathLike)):
            with open(os.fspath(model), 'rb') as source:
                copy.ParseFromString(source.read())
        else:
            raise TypeError('model must be an ONNX ModelProto or file path')
        self._check_embedded(copy)
        onnx.checker.check_model(copy)
        names = [weight.name for weight in copy.graph.initializer]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError('Initializers must have unique nonempty names')
        for weight in copy.graph.initializer:
            if weight.data_type == onnx.TensorProto.FLOAT:
                array = numpy_helper.to_array(weight)
                if array.size and np.isfinite(array).all():
                    weight.CopyFrom(numpy_helper.from_array(array, name=weight.name))
        self._native.saveModel(name, copy.SerializeToString(), tolerance)

    def loadModel(self, model_name):
        """Restore and return an ONNX ModelProto; missing/corrupt stores raise.

        The returned graph can be passed to singa.sonnx.SONNXModel, subject to
        SINGA's operator support. Loading does not write an exported ONNX file.
        """
        import onnx
        payload = self._native.loadModel(self._name(model_name))
        model = onnx.load_model_from_string(payload)
        onnx.checker.check_model(model)
        return model
