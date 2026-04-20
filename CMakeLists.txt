cmake_minimum_required(VERSION 3.14)
project(tensor_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

add_library(volta_sim
    src/volta_tensor.cpp
)
target_include_directories(volta_sim PUBLIC include)

add_library(ampere_sim
    src/ampere_tensor.cpp
)
target_include_directories(ampere_sim PUBLIC include)

add_library(hopper_sim
    src/hopper_tensor.cpp
)
target_include_directories(hopper_sim PUBLIC include)

add_library(blackwell_sim
    src/blackwell_tensor.cpp
)
target_include_directories(blackwell_sim PUBLIC include)

add_executable(test_volta
    tests/test_volta.cpp
)
target_link_libraries(test_volta PRIVATE volta_sim)
target_include_directories(test_volta PRIVATE include)

add_executable(test_ampere
    tests/test_ampere.cpp
)
target_link_libraries(test_ampere PRIVATE ampere_sim)
target_include_directories(test_ampere PRIVATE include)

add_executable(test_hopper
    tests/test_hopper.cpp
)
target_link_libraries(test_hopper PRIVATE hopper_sim)
target_include_directories(test_hopper PRIVATE include)

add_executable(test_blackwell
    tests/test_blackwell.cpp
)
target_link_libraries(test_blackwell PRIVATE blackwell_sim)
target_include_directories(test_blackwell PRIVATE include)

add_executable(tensor_sim
    main.cpp
)
target_link_libraries(tensor_sim PRIVATE volta_sim ampere_sim hopper_sim blackwell_sim)
target_include_directories(tensor_sim PRIVATE include)
target_compile_definitions(tensor_sim PRIVATE
    TENSOR_SIM_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MINGW)
    target_link_options(tensor_sim PRIVATE -static-libgcc -static-libstdc++)
endif()

enable_testing()
add_test(NAME volta_tests     COMMAND test_volta)
add_test(NAME ampere_tests    COMMAND test_ampere)
add_test(NAME hopper_tests    COMMAND test_hopper)
add_test(NAME blackwell_tests COMMAND test_blackwell)
