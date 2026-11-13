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

#include "singa/io/neurstore.h"
#include "neurstore/compress/compressor.h"
#include "neurstore/compress/utils/matrix_utils.h"
#include "neurstore/utils/storage_io.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void validateName(const std::string& name) {
    if (name.empty() || name == "." || name == ".." ||
        name.find_first_of("/\\") != std::string::npos ||
        name.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid model name");
}

std::vector<uint8_t> readFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open " + path.string());
    auto end = in.tellg();
    if (end < 0) throw std::runtime_error("Cannot size " + path.string());
    std::vector<uint8_t> data(static_cast<size_t>(end));
    in.seekg(0);
    if (!data.empty() && !in.read(reinterpret_cast<char*>(data.data()), data.size()))
        throw std::runtime_error("Cannot read " + path.string());
    return data;
}

// Legacy files contain native POD fields. Check every boundary before parsing.
struct Cursor {
    const uint8_t* data;
    size_t size, pos = 0;
    explicit Cursor(const std::vector<uint8_t>& bytes)
        : data(bytes.data()), size(bytes.size()) {}
    template<class T> T read() {
        T value;
        const auto* p = take(sizeof(T));
        std::memcpy(&value, p, sizeof(T));
        return value;
    }
    const uint8_t* take(size_t count) {
        if (count > size - pos) throw std::runtime_error("Truncated storage file");
        const auto* p = data + pos;
        pos += count;
        return p;
    }
    std::string text() {
        auto count = read<uint32_t>();
        return std::string(reinterpret_cast<const char*>(take(count)), count);
    }
    void done() {
        if (pos != size) throw std::runtime_error("Unexpected trailing storage data");
    }
};

ModelStructure readStructure(const fs::path& path) {
    auto bytes = readFile(path);
    Cursor c(bytes);
    auto length = c.read<uint32_t>();
    const auto* data = c.take(length);
    ModelStructure result(std::vector<uint8_t>(data, data + length), {});
    auto count = c.read<uint32_t>();
    if (count > (c.size - c.pos) / 4) throw std::runtime_error("Invalid name count");
    for (uint32_t i = 0; i < count; ++i) result.names.push_back(c.text());
    c.done();
    return result;
}

void validatePacket(const std::vector<uint8_t>& bytes) {
    Cursor c(bytes);
    double scale = c.read<double>(), zero = c.read<double>();
    int original = c.read<int>(), bits = c.read<int>();
    auto base = c.read<int64_t>(), index = c.read<int64_t>();
    auto nbytes = c.read<int64_t>(), n = c.read<int64_t>();
    size_t rank = c.read<size_t>();
    if (!std::isfinite(scale) || !std::isfinite(zero) || scale < 0 ||
        (original != 16 && original != 32 && original != 64) ||
        bits < 1 || bits > 32 || base < 0 || index < 0 || index > INT32_MAX ||
        n <= 0 || n > INT32_MAX || nbytes != (n * bits + 7) / 8 ||
        rank > (c.size - c.pos) / sizeof(int))
        throw std::runtime_error("Invalid tensor packet metadata");
    int64_t product = 1;
    for (size_t i = 0; i < rank; ++i) {
        int dim = c.read<int>();
        if (dim <= 0 || product > INT32_MAX / dim)
            throw std::runtime_error("Invalid tensor shape");
        product *= dim;
    }
    if (product != n) throw std::runtime_error("Tensor shape/element mismatch");
    c.take(static_cast<size_t>(nbytes));
    c.done();
}

TensorPage readPage(const fs::path& path) {
    auto bytes = readFile(path);
    Cursor c(bytes);
    auto count = c.read<uint32_t>();
    if (count > (c.size - c.pos) / 4) throw std::runtime_error("Invalid packet count");
    std::vector<std::shared_ptr<TensorPacket>> packets;
    std::vector<std::string> names;
    for (uint32_t i = 0; i < count; ++i) {
        auto length = c.read<uint32_t>();
        const auto* p = c.take(length);
        std::vector<uint8_t> packet(p, p + length);
        validatePacket(packet);
        packets.push_back(DeltaQuantPacket::deserialize(packet));
    }
    std::set<std::string> unique;
    for (uint32_t i = 0; i < count; ++i) {
        auto name = c.text();
        if (name.empty() || !unique.insert(name).second)
            throw std::runtime_error("Invalid or duplicate weight name");
        names.push_back(std::move(name));
    }
    c.done();
    return TensorPage(packets, names);
}
}

namespace singa {
struct Neurstore::Impl {
    fs::path directory;
    std::mutex mutex;
    explicit Impl(const std::string& path) : directory(path) {
        if (path.empty() || path.find('\0') != std::string::npos)
            throw std::invalid_argument("Invalid store directory");
        fs::create_directories(directory);
    }
};
Neurstore::Neurstore(const std::string& path) : impl_(new Impl(path)) {}
Neurstore::~Neurstore() = default;

void Neurstore::saveModel(const std::string& name, const std::string& bytes,
                         double tolerance) {
    validateName(name);
    if (!std::isfinite(tolerance) || tolerance <= 0)
        throw std::invalid_argument("tolerance must be positive and finite");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto structurePath = impl_->directory / (name + ".structure");
    const auto pagePath = impl_->directory / (name + ".tensors");
    if (fs::exists(structurePath) || fs::exists(pagePath))
        throw std::invalid_argument("Model already exists: " + name);
    onnx::ModelProto model;
    if (!model.ParseFromString(bytes)) throw std::invalid_argument("Invalid ONNX bytes");
    std::vector<onnx::TensorProto> weights, retained;
    std::set<std::string> namesSeen;
    for (const auto& tensor : model.graph().initializer()) {
        if (tensor.name().empty() || !namesSeen.insert(tensor.name()).second)
            throw std::invalid_argument("Invalid or duplicate initializer name");
        if (tensor.data_location() == onnx::TensorProto::EXTERNAL || tensor.external_data_size())
            throw std::invalid_argument("External ONNX weights must be embedded first");
        if (tensor.data_type() != onnx::TensorProto::FLOAT || tensor.raw_data().empty()) {
            retained.push_back(tensor);
            continue;
        }
        int64_t n = 1;
        for (auto d : tensor.dims()) {
            if (d < 0 || d > INT32_MAX || (d && n > INT32_MAX / d))
                throw std::invalid_argument("Unsupported tensor dimensions");
            n *= d;
        }
        if (tensor.raw_data().size() != static_cast<size_t>(n) * sizeof(float))
            throw std::invalid_argument("Invalid FLOAT tensor data length");
        bool finite = true;
        for (int64_t i = 0; i < n; ++i) {
            float value;
            std::memcpy(&value, tensor.raw_data().data() + i * sizeof(float), sizeof(float));
            finite = finite && std::isfinite(value);
        }
        if (n && finite) weights.push_back(tensor);
        else retained.push_back(tensor);
    }
    model.mutable_graph()->clear_initializer();
    for (const auto& tensor : retained) *model.mutable_graph()->add_initializer() = tensor;
    std::string structureBytes;
    if (!model.SerializeToString(&structureBytes)) throw std::runtime_error("ONNX serialization failed");
    ModelStructure structure(std::vector<uint8_t>(structureBytes.begin(), structureBytes.end()), {});
    auto cache = std::make_shared<IndexCacheManager>(impl_->directory.string());
    Compressor compressor(tolerance, true, 8, cache);
    std::unordered_map<int, std::vector<PendingInsertion>> pending;
    std::vector<std::shared_ptr<TensorPacket>> packets;
    std::vector<std::string> names;
    // Sequential first version: native errors always reach the SWIG caller.
    // Pending bases remain invisible until the whole first pass finishes.
    for (const auto& weight : weights) {
        auto tensor = std::make_shared<TensorF64>(MatrixUtils::tensorProto2TensorF64(weight));
        packets.push_back(compressor.compressTensor(tensor, packets.size(), pending));
        names.push_back(weight.name());
    }
    for (auto& group : pending) for (auto& insertion : group.second) {
        auto index = cache->get(insertion.dimension, insertion.index_id);
        auto id = index->insert(insertion.tensor, insertion.scale, insertion.zero_point);
        packets[insertion.tensor_sequence_num]->setBaseTensorId(id);
        cache->release(index, insertion.dimension, insertion.index_id, true);
    }
    cache->clearCache();
    const auto structureData = structure.serialize();
    const auto pageData = TensorPage(packets, names).serialize();
    // The structure file is published last. A partial pair is never loadable.
    atomicStorageWrite(pagePath.string(), pageData);
    try { atomicStorageWrite(structurePath.string(), structureData); }
    catch (...) { fs::remove(pagePath); throw; }
}

std::string Neurstore::loadModel(const std::string& name) {
    validateName(name);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto structure = readStructure(impl_->directory / (name + ".structure"));
    auto page = readPage(impl_->directory / (name + ".tensors"));
    onnx::ModelProto model;
    if (!model.ParseFromArray(structure.serialized_model_structure.data(),
                              structure.serialized_model_structure.size()))
        throw std::runtime_error("Invalid stored ONNX structure");
    auto cache = std::make_shared<IndexCacheManager>(impl_->directory.string());
    DeltaQuantCompress codec(0.0, true, 8);
    std::set<std::string> names;
    for (const auto& tensor : model.graph().initializer()) names.insert(tensor.name());
    for (uint32_t i = 0; i < page.nTensors(); ++i) {
        auto packet = page.getTensorPacket(i);
        if (!names.insert(page.getName(i)).second)
            throw std::runtime_error("Duplicate stored initializer name");
        auto indexPath = impl_->directory / (std::to_string(packet->getDimension()) + "_" +
                          std::to_string(packet->getIndexId()) + ".index");
        if (!fs::is_regular_file(indexPath))
            throw std::runtime_error("Missing shared index: " + indexPath.string());
        auto index = cache->get(packet->getDimension(), packet->getIndexId());
        auto base = index->retrieve(packet->getBaseTensorId());
        if (base.size() != packet->getDimension() || !base.array().isFinite().all())
            throw std::runtime_error("Invalid shared base tensor");
        auto tensor = codec.decompress(base, packet);
        cache->release(index, packet->getDimension(), packet->getIndexId(), false);
        auto* output = model.mutable_graph()->add_initializer();
        output->set_name(page.getName(i));
        output->set_data_type(onnx::TensorProto::FLOAT);
        for (auto d : packet->getShape()) output->add_dims(d);
        auto data = tensor->getTensor().cast<float>().eval();
        output->set_raw_data(data.data(), data.size() * sizeof(float));
    }
    std::string result;
    if (!model.SerializeToString(&result)) throw std::runtime_error("ONNX serialization failed");
    return result;
}
}
