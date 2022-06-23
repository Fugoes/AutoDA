cmake_minimum_required(VERSION 3.16)
project(autoda)

set(CMAKE_CXX_STANDARD 17)
set(WFLAGS "-Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-variable")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WFLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WFLAGS} -flto")

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)

link_directories(${HDF5_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set(TensorFlow_LIBRARIES tensorflow tensorflow_framework)

include_directories("src")

add_library(autoda_library
        src/models.cpp
        src/config.cpp
        src/ops.cpp
        src/dag.cpp
        src/tac.cpp
        src/alg.cpp
        src/limited_random_alg.cpp)
target_link_libraries(autoda_library
        ${Boost_LIBRARIES}
        ${TensorFlow_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
        pthread
        -static-libgcc
        -static-libstdc++)
target_precompile_headers(autoda_library PRIVATE src/prelude.hpp)

add_executable(autoda bin/autoda.cpp)
target_link_libraries(autoda PRIVATE autoda_library)
target_precompile_headers(autoda REUSE_FROM autoda_library)

add_executable(autoda_ablation bin/autoda_ablation.cpp)
target_link_libraries(autoda_ablation PRIVATE autoda_library)
target_precompile_headers(autoda_ablation REUSE_FROM autoda_library)

add_executable(test_dataset bin/test_dataset.cpp)
target_link_libraries(test_dataset PRIVATE autoda_library)
target_precompile_headers(test_dataset REUSE_FROM autoda_library)

add_executable(test_dag bin/test_dag.cpp)
target_link_libraries(test_dag PRIVATE autoda_library)
target_precompile_headers(test_dag REUSE_FROM autoda_library)

add_executable(test_dag_compile bin/test_dag_compile.cpp)
target_link_libraries(test_dag_compile PRIVATE autoda_library)
target_precompile_headers(test_dag_compile REUSE_FROM autoda_library)

add_executable(test_attacker bin/test_attacker.cpp)
target_link_libraries(test_attacker PRIVATE autoda_library)
target_precompile_headers(test_attacker REUSE_FROM autoda_library)

add_executable(test_boundary bin/test_boundary.cpp)
target_link_libraries(test_boundary PRIVATE autoda_library)
target_precompile_headers(test_boundary REUSE_FROM autoda_library)

add_executable(test_eigen bin/test_eigen.cpp)

add_executable(test_model bin/test_model.cpp)
target_link_libraries(test_model PRIVATE autoda_library)
target_precompile_headers(test_model REUSE_FROM autoda_library)

add_executable(test_random bin/test_random.cpp)
target_link_libraries(test_random PRIVATE autoda_library)
target_precompile_headers(test_random REUSE_FROM autoda_library)
