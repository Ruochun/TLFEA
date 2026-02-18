# Examples/Demos CMakeLists.txt

# Add the basic infrastructure test example (Nesterov solver)
add_executable(test_feat10_nesterov
    test_feat10_nesterov.cc
)

target_link_libraries(test_feat10_nesterov
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_feat10_nesterov PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_feat10_nesterov PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the Newton solver test example
add_executable(test_feat10_newton
    test_feat10_newton.cc
)

target_link_libraries(test_feat10_newton
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_feat10_newton PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_feat10_newton PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the beam simulation example (Nesterov solver)
add_executable(beam_simulation
    beam_simulation.cc
)

target_link_libraries(beam_simulation
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(beam_simulation PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(beam_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the beam simulation example (Newton solver)
add_executable(beam_simulation_newton
    beam_simulation_newton.cc
)

target_link_libraries(beam_simulation_newton
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(beam_simulation_newton PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(beam_simulation_newton PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
