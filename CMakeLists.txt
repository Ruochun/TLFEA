cmake_minimum_required(VERSION 3.18)
project(TLFEA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

# Add MoPhiEssentials subdirectory
add_subdirectory(external/MoPhiEssentials)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# Collect source files
# Note: CONFIGURE_DEPENDS ensures CMake rechecks when files are added/removed
file(GLOB ELEMENT_SOURCES CONFIGURE_DEPENDS src/elements/*.cu)
file(GLOB SOLVER_SOURCES CONFIGURE_DEPENDS src/solvers/*.cu)
file(GLOB UTILS_SOURCES CONFIGURE_DEPENDS src/utils/*.cc)

# Create a library with all FEA components
add_library(tlfea_lib STATIC
    ${ELEMENT_SOURCES}
    ${SOLVER_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusparse
    mophi_essentials
)

target_include_directories(tlfea_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# Add the example executable
add_executable(test_feat10_nesterov
    examples/test_feat10_nesterov.cc
)

target_link_libraries(test_feat10_nesterov
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusparse
    mophi_essentials
)

target_include_directories(test_feat10_nesterov PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# Set the runtime path for the data directory
set_target_properties(test_feat10_nesterov PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the cantilever beam simulation example
add_executable(cantilever_beam_simulation
    examples/cantilever_beam_simulation.cc
)

target_link_libraries(cantilever_beam_simulation
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusparse
    mophi_essentials
)

target_include_directories(cantilever_beam_simulation PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(cantilever_beam_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the VTU beam simulation example
add_executable(vtu_beam_simulation
    examples/vtu_beam_simulation.cc
)

target_link_libraries(vtu_beam_simulation
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusparse
    mophi_essentials
)

target_include_directories(vtu_beam_simulation PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(vtu_beam_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
