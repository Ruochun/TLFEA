# Examples/Demos CMakeLists.txt

# Add the basic infrastructure test example
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

# Add the AdamW test example
add_executable(test_feat10_adamw
    test_feat10_adamw.cc
)

target_link_libraries(test_feat10_adamw
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_feat10_adamw PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_feat10_adamw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the beam simulation example
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

# Add the Newton solver beam demo
add_executable(beam_newton_demo
    beam_newton_demo.cc
)

target_link_libraries(beam_newton_demo
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(beam_newton_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(beam_newton_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
