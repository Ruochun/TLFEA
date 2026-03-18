cmake_minimum_required(VERSION 3.18)
project(TLFEA VERSION 0.1.0 LANGUAGES CXX CUDA)

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
    PUBLIC
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cusparse
        CUDA::cublas
    PRIVATE
        mophi_essentials
)

target_include_directories(tlfea_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tlfea>
)

# Add examples subdirectory
add_subdirectory(examples)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# ===========================================================================
# Installation policy
# ===========================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the compiled library and export its targets
install(TARGETS tlfea_lib
    EXPORT TLFEATargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install all public headers, preserving the src/ subdirectory structure
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tlfea
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh"
)

# Install bundled MoPhiEssentials headers alongside TLFEA headers so
# downstream consumers that use the installed package still have them.
install(DIRECTORY ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tlfea
    FILES_MATCHING
        PATTERN "*.h"
)

# Export targets file (for use from an install tree)
install(EXPORT TLFEATargets
    FILE TLFEATargets.cmake
    NAMESPACE TLFEA::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TLFEA
)

# Generate a version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TLFEAConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Generate the package config file from the template
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/TLFEAConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TLFEAConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TLFEA
)

# Install the config and version files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TLFEAConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TLFEAConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TLFEA
)
