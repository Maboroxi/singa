# Optional standalone storage core and SWIG binding. No database/inference deps.
find_path(NEURSTORE_EIGEN_INCLUDE_DIR Eigen/Core PATH_SUFFIXES eigen3)
if(NOT NEURSTORE_EIGEN_INCLUDE_DIR)
    message(FATAL_ERROR "Set NEURSTORE_EIGEN_INCLUDE_DIR to the Eigen include root")
endif()
find_package(OpenMP REQUIRED)
set(NS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(NS_GEN ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${NS_GEN})
add_custom_command(OUTPUT ${NS_GEN}/onnx.pb.cc ${NS_GEN}/onnx.pb.h
    COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${NS_GEN}
            -I ${NS_ROOT}/schema ${NS_ROOT}/schema/onnx.proto
    DEPENDS ${NS_ROOT}/schema/onnx.proto VERBATIM)
file(GLOB NS_METHODS ${NS_ROOT}/src/neurstore/compress/method/*.cpp)
file(GLOB NS_STRATEGIES ${NS_ROOT}/src/neurstore/compress/strategy/*.cpp)
file(GLOB NS_UTILS ${NS_ROOT}/src/neurstore/compress/utils/*.cpp)
add_library(singa_onnx_storage STATIC
    src/storage.cpp src/codec.cpp
    src/neurstore/cache/index_cache_manager.cpp
    src/neurstore/compress/index/quantized_hnsw_hnswlib.cpp
    src/neurstore/compress/index/router/single_index_router.cpp
    src/neurstore/utils/model.cpp src/neurstore/utils/tensor.cpp
    src/neurstore/utils/timer.cpp ${NS_METHODS} ${NS_STRATEGIES} ${NS_UTILS}
    ${NS_GEN}/onnx.pb.cc)
set_target_properties(singa_onnx_storage PROPERTIES CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_compile_options(singa_onnx_storage PRIVATE -mavx2 -mfma)
target_include_directories(singa_onnx_storage PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${NS_ROOT}/include ${NS_ROOT}/external/hnswlib
    ${NS_GEN} ${NEURSTORE_EIGEN_INCLUDE_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(singa_onnx_storage PUBLIC ${Protobuf_LIBRARIES} OpenMP::OpenMP_CXX)

if(USE_PYTHON)
    set(NS_PY ${CMAKE_BINARY_DIR}/python/singa)
    file(MAKE_DIRECTORY ${NS_PY})
    add_custom_command(OUTPUT ${NS_GEN}/neurstore_wrap.cxx ${NS_PY}/neurstore_wrap.py
        COMMAND ${SWIG_EXECUTABLE} -c++ -python -py3
            -I${PROJECT_SOURCE_DIR}/include -outdir ${NS_PY}
            -o ${NS_GEN}/neurstore_wrap.cxx ${PROJECT_SOURCE_DIR}/src/api/neurstore.i
        DEPENDS ${PROJECT_SOURCE_DIR}/src/api/neurstore.i
                ${PROJECT_SOURCE_DIR}/include/singa/io/neurstore.h VERBATIM)
    add_library(_neurstore_wrap MODULE ${NS_GEN}/neurstore_wrap.cxx)
    target_include_directories(_neurstore_wrap PRIVATE ${PROJECT_SOURCE_DIR}/include ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(_neurstore_wrap singa_onnx_storage ${PYTHON_LIBRARIES})
    set_target_properties(_neurstore_wrap PROPERTIES PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${NS_PY})
endif()
