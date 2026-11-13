// Adapted from NeurStore compressor.cpp; only the tensor codec is linked.
#include "neurstore/compress/compressor.h"

#include <omp.h>


#include "neurstore/compress/index/router/single_index_router.h"
#include "neurstore/compress/method/linear_quantization.h"
#include "neurstore/compress/strategy/standard_nbits_strategy.h"
#include "neurstore/compress/utils/matrix_utils.h"
#include "neurstore/utils/global.h"

/*********************************** C++ ***********************************/

Compressor::Compressor(
    double tolerance,
    bool dynamic,
    int default_quantized_bit_width,
    std::shared_ptr<IndexCacheManager> index_cache_manager
): tolerance_(tolerance),
   dynamic_(dynamic),
   default_quantized_bit_width_(default_quantized_bit_width),
   delta_quant_compress_(tolerance, dynamic, default_quantized_bit_width),
   selection_strategy_(std::make_shared<StandardNBitsStrategy>(tolerance, default_quantized_bit_width)),
   index_cache_manager_(std::move(index_cache_manager)) {
    if (!dynamic && default_quantized_bit_width <= 0) {
        throw std::invalid_argument("Compressor: default quantized bit width must be positive if dynamic is false");
    }
}

std::shared_ptr<TensorPacket> Compressor::compressTensor(
    const std::shared_ptr<TensorF64> &tensor,
    const uint tensor_index,
    std::unordered_map<
        int, std::vector<PendingInsertion>
    > &pending_insertion_tensors
) {
    // 8-bit quant
    TensorType::VectorFloat64 tensor_data = tensor->getTensor();
    TensorType::VectorUInt8 quantized_tensor;
    double scale;
    double zero_point;
    LinearQuantization::linearAsymmetricQuantize(
        tensor_data,
        8,
        quantized_tensor,
        scale,
        zero_point
    );
    // 找相似base向量
    std::shared_ptr<VectorFloat64Router> router = index_cache_manager_->getRouter(tensor->getDimension());
    int index_id = router->route(tensor_data);
    auto index = index_cache_manager_->get(tensor->getDimension(), index_id);

    auto representative_id = index->query(quantized_tensor, scale, zero_point);

    // if the dimension of the tensor is 1, we directly insert it into the index
    // TODO: this is a special case for 1D tensors, we should fix this corner case in the future
    if (tensor->getDimension() == 1) {
        representative_id = -1;
    }

    TensorType::VectorFloat64 base_tensor;
    // negative，新建索引
    if (representative_id == -1) {
        TensorType::VectorFloat64 tensor_dequantized;
        LinearQuantization::linearAsymmetricDequantize(
            quantized_tensor,
            scale,
            zero_point,
            tensor_dequantized
        );
        int dimension = tensor->getDimension();
        pending_insertion_tensors[index_id].emplace_back(
            dimension,
            index_id,
            std::move(quantized_tensor),
            scale,
            zero_point,
            tensor_index
        );
        base_tensor = tensor_dequantized;
        router->update(
            index_id, tensor_data, static_cast<int>(index->size())
        );
    } else {
        base_tensor = index->retrieve(representative_id);
        // 如果需要当前张量成为新的代表张量
        if (selection_strategy_->selectAsRepresentative(base_tensor, tensor_data)) {
            TensorType::VectorFloat64 tensor_dequantized;
            LinearQuantization::linearAsymmetricDequantize(
                quantized_tensor,
                scale,
                zero_point,
                tensor_dequantized
            );
            int dimension = tensor->getDimension();
            pending_insertion_tensors[index_id].emplace_back(
                dimension,
                index_id,
                std::move(quantized_tensor),
                scale,
                zero_point,
                tensor_index
            );
            base_tensor = tensor_dequantized;
            representative_id = -1;
            router->update(
                index_id, tensor_data, static_cast<int>(index->size())
            );
        }
    }
    index_cache_manager_->release(index, tensor->getDimension(), index_id, false);

    std::shared_ptr<TensorPacket> packet = delta_quant_compress_.compress(
        base_tensor,
        tensor_data,
        tensor->getShape()
    );
    packet->setBaseTensorId(representative_id);
    packet->setIndexId(index_id);
    return packet;
}

