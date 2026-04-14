# CMake generated Testfile for 
# Source directory: G:/tensorscope/tensor_sim
# Build directory: G:/tensorscope/tensor_sim/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(volta_tests "G:/tensorscope/tensor_sim/build/test_volta.exe")
set_tests_properties(volta_tests PROPERTIES  _BACKTRACE_TRIPLES "G:/tensorscope/tensor_sim/CMakeLists.txt;76;add_test;G:/tensorscope/tensor_sim/CMakeLists.txt;0;")
add_test(ampere_tests "G:/tensorscope/tensor_sim/build/test_ampere.exe")
set_tests_properties(ampere_tests PROPERTIES  _BACKTRACE_TRIPLES "G:/tensorscope/tensor_sim/CMakeLists.txt;77;add_test;G:/tensorscope/tensor_sim/CMakeLists.txt;0;")
add_test(hopper_tests "G:/tensorscope/tensor_sim/build/test_hopper.exe")
set_tests_properties(hopper_tests PROPERTIES  _BACKTRACE_TRIPLES "G:/tensorscope/tensor_sim/CMakeLists.txt;78;add_test;G:/tensorscope/tensor_sim/CMakeLists.txt;0;")
add_test(blackwell_tests "G:/tensorscope/tensor_sim/build/test_blackwell.exe")
set_tests_properties(blackwell_tests PROPERTIES  _BACKTRACE_TRIPLES "G:/tensorscope/tensor_sim/CMakeLists.txt;79;add_test;G:/tensorscope/tensor_sim/CMakeLists.txt;0;")
