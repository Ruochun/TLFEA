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
    CUDA::cublas
    mophi_essentials
)

target_include_directories(tlfea_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# Add examples subdirectory
add_subdirectory(examples)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
