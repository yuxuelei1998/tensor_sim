default_target: all
.PHONY : default_target

.NOTPARALLEL:

.SUFFIXES:
% : %,v
% : RCS/%
% : RCS/%,v
% : SCCS/s.%
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list
$(VERBOSE)MAKESILENT = -s

$(VERBOSE).SILENT:

cmake_force:
.PHONY : cmake_force

SHELL = cmd.exe

CMAKE_COMMAND = "G:\\CLion 2025.2.3\\bin\\cmake\\win\\x64\\bin\\cmake.exe"
RM = "G:\\CLion 2025.2.3\\bin\\cmake\\win\\x64\\bin\\cmake.exe" -E rm -f

EQUALS = =

CMAKE_SOURCE_DIR = G:\tensor_sim

CMAKE_BINARY_DIR = G:\tensor_sim\build

test:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running tests..."
	"G:\CLion 2025.2.3\bin\cmake\win\x64\bin\ctest.exe" $(ARGS)
.PHONY : test
test/fast: test
.PHONY : test/fast

edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "No interactive CMake dialog available..."
	"G:\CLion 2025.2.3\bin\cmake\win\x64\bin\cmake.exe" -E echo "No interactive CMake dialog available."
.PHONY : edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	"G:\CLion 2025.2.3\bin\cmake\win\x64\bin\cmake.exe" --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start G:\tensor_sim\build\CMakeFiles G:\tensor_sim\build\\CMakeFiles\progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start G:\tensor_sim\build\CMakeFiles 0
.PHONY : all

clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 clean
.PHONY : clean

clean/fast: clean
.PHONY : clean/fast

preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 preinstall
.PHONY : preinstall

preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 preinstall
.PHONY : preinstall/fast

depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles\Makefile.cmake 1
.PHONY : depend

volta_sim: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 volta_sim
.PHONY : volta_sim

volta_sim/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\volta_sim.dir\build.make CMakeFiles/volta_sim.dir/build
.PHONY : volta_sim/fast

ampere_sim: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 ampere_sim
.PHONY : ampere_sim

ampere_sim/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\ampere_sim.dir\build.make CMakeFiles/ampere_sim.dir/build
.PHONY : ampere_sim/fast

hopper_sim: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 hopper_sim
.PHONY : hopper_sim

hopper_sim/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\hopper_sim.dir\build.make CMakeFiles/hopper_sim.dir/build
.PHONY : hopper_sim/fast

blackwell_sim: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 blackwell_sim
.PHONY : blackwell_sim

blackwell_sim/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\blackwell_sim.dir\build.make CMakeFiles/blackwell_sim.dir/build
.PHONY : blackwell_sim/fast

test_volta: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 test_volta
.PHONY : test_volta

test_volta/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_volta.dir\build.make CMakeFiles/test_volta.dir/build
.PHONY : test_volta/fast

test_ampere: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 test_ampere
.PHONY : test_ampere

test_ampere/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_ampere.dir\build.make CMakeFiles/test_ampere.dir/build
.PHONY : test_ampere/fast

test_hopper: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 test_hopper
.PHONY : test_hopper

test_hopper/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_hopper.dir\build.make CMakeFiles/test_hopper.dir/build
.PHONY : test_hopper/fast

test_blackwell: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 test_blackwell
.PHONY : test_blackwell

test_blackwell/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_blackwell.dir\build.make CMakeFiles/test_blackwell.dir/build
.PHONY : test_blackwell/fast

tensor_sim: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 tensor_sim
.PHONY : tensor_sim

tensor_sim/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\tensor_sim.dir\build.make CMakeFiles/tensor_sim.dir/build
.PHONY : tensor_sim/fast

main.obj: main.cpp.obj
.PHONY : main.obj

main.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\tensor_sim.dir\build.make CMakeFiles/tensor_sim.dir/main.cpp.obj
.PHONY : main.cpp.obj

main.i: main.cpp.i
.PHONY : main.i

main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\tensor_sim.dir\build.make CMakeFiles/tensor_sim.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s
.PHONY : main.s

main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\tensor_sim.dir\build.make CMakeFiles/tensor_sim.dir/main.cpp.s
.PHONY : main.cpp.s

src/ampere_tensor.obj: src/ampere_tensor.cpp.obj
.PHONY : src/ampere_tensor.obj

src/ampere_tensor.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\ampere_sim.dir\build.make CMakeFiles/ampere_sim.dir/src/ampere_tensor.cpp.obj
.PHONY : src/ampere_tensor.cpp.obj

src/ampere_tensor.i: src/ampere_tensor.cpp.i
.PHONY : src/ampere_tensor.i

src/ampere_tensor.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\ampere_sim.dir\build.make CMakeFiles/ampere_sim.dir/src/ampere_tensor.cpp.i
.PHONY : src/ampere_tensor.cpp.i

src/ampere_tensor.s: src/ampere_tensor.cpp.s
.PHONY : src/ampere_tensor.s

src/ampere_tensor.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\ampere_sim.dir\build.make CMakeFiles/ampere_sim.dir/src/ampere_tensor.cpp.s
.PHONY : src/ampere_tensor.cpp.s

src/blackwell_tensor.obj: src/blackwell_tensor.cpp.obj
.PHONY : src/blackwell_tensor.obj

src/blackwell_tensor.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\blackwell_sim.dir\build.make CMakeFiles/blackwell_sim.dir/src/blackwell_tensor.cpp.obj
.PHONY : src/blackwell_tensor.cpp.obj

src/blackwell_tensor.i: src/blackwell_tensor.cpp.i
.PHONY : src/blackwell_tensor.i

src/blackwell_tensor.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\blackwell_sim.dir\build.make CMakeFiles/blackwell_sim.dir/src/blackwell_tensor.cpp.i
.PHONY : src/blackwell_tensor.cpp.i

src/blackwell_tensor.s: src/blackwell_tensor.cpp.s
.PHONY : src/blackwell_tensor.s

src/blackwell_tensor.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\blackwell_sim.dir\build.make CMakeFiles/blackwell_sim.dir/src/blackwell_tensor.cpp.s
.PHONY : src/blackwell_tensor.cpp.s

src/hopper_tensor.obj: src/hopper_tensor.cpp.obj
.PHONY : src/hopper_tensor.obj

src/hopper_tensor.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\hopper_sim.dir\build.make CMakeFiles/hopper_sim.dir/src/hopper_tensor.cpp.obj
.PHONY : src/hopper_tensor.cpp.obj

src/hopper_tensor.i: src/hopper_tensor.cpp.i
.PHONY : src/hopper_tensor.i

src/hopper_tensor.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\hopper_sim.dir\build.make CMakeFiles/hopper_sim.dir/src/hopper_tensor.cpp.i
.PHONY : src/hopper_tensor.cpp.i

src/hopper_tensor.s: src/hopper_tensor.cpp.s
.PHONY : src/hopper_tensor.s

src/hopper_tensor.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\hopper_sim.dir\build.make CMakeFiles/hopper_sim.dir/src/hopper_tensor.cpp.s
.PHONY : src/hopper_tensor.cpp.s

src/volta_tensor.obj: src/volta_tensor.cpp.obj
.PHONY : src/volta_tensor.obj

src/volta_tensor.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\volta_sim.dir\build.make CMakeFiles/volta_sim.dir/src/volta_tensor.cpp.obj
.PHONY : src/volta_tensor.cpp.obj

src/volta_tensor.i: src/volta_tensor.cpp.i
.PHONY : src/volta_tensor.i

src/volta_tensor.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\volta_sim.dir\build.make CMakeFiles/volta_sim.dir/src/volta_tensor.cpp.i
.PHONY : src/volta_tensor.cpp.i

src/volta_tensor.s: src/volta_tensor.cpp.s
.PHONY : src/volta_tensor.s

src/volta_tensor.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\volta_sim.dir\build.make CMakeFiles/volta_sim.dir/src/volta_tensor.cpp.s
.PHONY : src/volta_tensor.cpp.s

tests/test_ampere.obj: tests/test_ampere.cpp.obj
.PHONY : tests/test_ampere.obj

tests/test_ampere.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_ampere.dir\build.make CMakeFiles/test_ampere.dir/tests/test_ampere.cpp.obj
.PHONY : tests/test_ampere.cpp.obj

tests/test_ampere.i: tests/test_ampere.cpp.i
.PHONY : tests/test_ampere.i

tests/test_ampere.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_ampere.dir\build.make CMakeFiles/test_ampere.dir/tests/test_ampere.cpp.i
.PHONY : tests/test_ampere.cpp.i

tests/test_ampere.s: tests/test_ampere.cpp.s
.PHONY : tests/test_ampere.s

tests/test_ampere.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_ampere.dir\build.make CMakeFiles/test_ampere.dir/tests/test_ampere.cpp.s
.PHONY : tests/test_ampere.cpp.s

tests/test_blackwell.obj: tests/test_blackwell.cpp.obj
.PHONY : tests/test_blackwell.obj

tests/test_blackwell.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_blackwell.dir\build.make CMakeFiles/test_blackwell.dir/tests/test_blackwell.cpp.obj
.PHONY : tests/test_blackwell.cpp.obj

tests/test_blackwell.i: tests/test_blackwell.cpp.i
.PHONY : tests/test_blackwell.i

tests/test_blackwell.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_blackwell.dir\build.make CMakeFiles/test_blackwell.dir/tests/test_blackwell.cpp.i
.PHONY : tests/test_blackwell.cpp.i

tests/test_blackwell.s: tests/test_blackwell.cpp.s
.PHONY : tests/test_blackwell.s

tests/test_blackwell.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_blackwell.dir\build.make CMakeFiles/test_blackwell.dir/tests/test_blackwell.cpp.s
.PHONY : tests/test_blackwell.cpp.s

tests/test_hopper.obj: tests/test_hopper.cpp.obj
.PHONY : tests/test_hopper.obj

tests/test_hopper.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_hopper.dir\build.make CMakeFiles/test_hopper.dir/tests/test_hopper.cpp.obj
.PHONY : tests/test_hopper.cpp.obj

tests/test_hopper.i: tests/test_hopper.cpp.i
.PHONY : tests/test_hopper.i

tests/test_hopper.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_hopper.dir\build.make CMakeFiles/test_hopper.dir/tests/test_hopper.cpp.i
.PHONY : tests/test_hopper.cpp.i

tests/test_hopper.s: tests/test_hopper.cpp.s
.PHONY : tests/test_hopper.s

tests/test_hopper.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_hopper.dir\build.make CMakeFiles/test_hopper.dir/tests/test_hopper.cpp.s
.PHONY : tests/test_hopper.cpp.s

tests/test_volta.obj: tests/test_volta.cpp.obj
.PHONY : tests/test_volta.obj

tests/test_volta.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_volta.dir\build.make CMakeFiles/test_volta.dir/tests/test_volta.cpp.obj
.PHONY : tests/test_volta.cpp.obj

tests/test_volta.i: tests/test_volta.cpp.i
.PHONY : tests/test_volta.i

tests/test_volta.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_volta.dir\build.make CMakeFiles/test_volta.dir/tests/test_volta.cpp.i
.PHONY : tests/test_volta.cpp.i

tests/test_volta.s: tests/test_volta.cpp.s
.PHONY : tests/test_volta.s

tests/test_volta.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\test_volta.dir\build.make CMakeFiles/test_volta.dir/tests/test_volta.cpp.s
.PHONY : tests/test_volta.cpp.s

help:
	@echo The following are some of the valid targets for this Makefile:
	@echo ... all (the default if no target is provided)
	@echo ... clean
	@echo ... depend
	@echo ... edit_cache
	@echo ... rebuild_cache
	@echo ... test
	@echo ... ampere_sim
	@echo ... blackwell_sim
	@echo ... hopper_sim
	@echo ... tensor_sim
	@echo ... test_ampere
	@echo ... test_blackwell
	@echo ... test_hopper
	@echo ... test_volta
	@echo ... volta_sim
	@echo ... main.obj
	@echo ... main.i
	@echo ... main.s
	@echo ... src/ampere_tensor.obj
	@echo ... src/ampere_tensor.i
	@echo ... src/ampere_tensor.s
	@echo ... src/blackwell_tensor.obj
	@echo ... src/blackwell_tensor.i
	@echo ... src/blackwell_tensor.s
	@echo ... src/hopper_tensor.obj
	@echo ... src/hopper_tensor.i
	@echo ... src/hopper_tensor.s
	@echo ... src/volta_tensor.obj
	@echo ... src/volta_tensor.i
	@echo ... src/volta_tensor.s
	@echo ... tests/test_ampere.obj
	@echo ... tests/test_ampere.i
	@echo ... tests/test_ampere.s
	@echo ... tests/test_blackwell.obj
	@echo ... tests/test_blackwell.i
	@echo ... tests/test_blackwell.s
	@echo ... tests/test_hopper.obj
	@echo ... tests/test_hopper.i
	@echo ... tests/test_hopper.s
	@echo ... tests/test_volta.obj
	@echo ... tests/test_volta.i
	@echo ... tests/test_volta.s
.PHONY : help

cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles\Makefile.cmake 0
.PHONY : cmake_check_build_system