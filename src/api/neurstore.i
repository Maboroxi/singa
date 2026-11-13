/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

%module neurstore_wrap
%{
#include "singa/io/neurstore.h"
%}
%include "std_string.i"
%include "exception.i"

%typemap(in) const std::string &onnx_bytes (std::string temporary) {
    if (!PyBytes_Check($input)) {
        SWIG_exception_fail(SWIG_TypeError, "ONNX payload must be bytes");
    }
    temporary.assign(PyBytes_AS_STRING($input), PyBytes_GET_SIZE($input));
    $1 = &temporary;
}
%typemap(freearg) const std::string &onnx_bytes "";
%typemap(out) std::string {
    $result = PyBytes_FromStringAndSize($1.data(), $1.size());
}
%exception {
    try { $action }
    catch (const std::invalid_argument& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    }
    catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}
%include "singa/io/neurstore.h"
