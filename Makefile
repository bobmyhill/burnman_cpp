# Copyright (c) 2025 Benedict Heinen
#
# This file is part of burnman_cpp and is licensed under the
# GNU General Public License v3.0 or later. See the LICENSE file
# or <https://www.gnu.org/licenses/> for details.
#
# burnman_cpp is based on BurnMan: <https://geodynamics.github.io/burnman/>
#
# ------------------- END OF LICENSE SECTION -----------------
# The native sources implement a library; Catch2 supplies main() for tests.
TARGET_LIBRARY := libburnman.a
TEST_EXEC := run_tests
BENCHMARK_EXEC := run_benchmarks

# Compiler
CXX := g++

# Directories
CWD := $(shell pwd)
SRC_DIR := $(CWD)/src
INCLUDE_DIR := $(CWD)/include
BUILD_DIR := $(CWD)/build
BIN_DIR := $(CWD)/bin
TEST_DIR := $(CWD)/tests
BENCHMARK_DIR := $(CWD)/benchmarks
INCLUDE_DIR_TEST := $(TEST_DIR)/include

# For GSL, etc. (Change if non-standard)
# Only defined here so they can be changed
# both should be on the system path anyway
# Eigen on MacOS installed via brew is in /opt/homebrew/include/eigen3
# GSL on MacOS installed via brew is in /opt/homebrew/lib
EXTRA_INCLUDE ?= $(wildcard /opt/homebrew/include /opt/homebrew/include/eigen3 /opt/homebrew/opt/gsl/include /usr/local/include /usr/local/include/eigen3 /usr/include/eigen3)
EXTRA_LIB ?= $(wildcard /usr/local/lib /opt/homebrew/lib)
LDFLAGS_COMMON := -lgsl -lgslcblas -lcddgmp -lgmp -lnlopt -lm
LDFLAGS_TEST := -lCatch2Main -lCatch2

# Expand to paths
INCLUDE_FLAGS_COMMON := -I$(INCLUDE_DIR) $(addprefix -isystem , $(EXTRA_INCLUDE))
INCLUDE_FLAGS_TEST := -I$(INCLUDE_DIR_TEST)
LIB_PATHS := $(addprefix -L, $(EXTRA_LIB))

# Compiler flags
CXXFLAGS := -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++17
CPPFLAGS := -MMD -MP $(INCLUDE_FLAGS_COMMON)
WARNINGS_AS_ERRORS ?= 0
ifeq ($(WARNINGS_AS_ERRORS), 1)
  CXXFLAGS += -Werror
endif

# Native builds target this machine by default. An older SDK default can be
# incompatible with locally installed Homebrew libraries. Cross-builds may
# specify MACOSX_DEPLOYMENT_TARGET explicitly.
ifeq ($(shell uname -s), Darwin)
  MACOSX_DEPLOYMENT_TARGET ?= $(shell sw_vers -productVersion | cut -d. -f1,2)
  CXXFLAGS += -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
  LDFLAGS_COMMON += -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
endif

# Build type specific flags
DEBUG_FLAGS = -g -Og
RELEASE_FLAGS = -O3 -DNDEBUG
TEST_FLAGS = -g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined

# Select Mode (default: release)
BUILD_MODE ?= release
ifeq ($(filter $(BUILD_MODE),release debug test),)
  $(error BUILD_MODE must be release, debug or test)
endif
ifeq ($(BUILD_MODE), debug)
  CXXFLAGS += $(DEBUG_FLAGS)
  BIN_SUFFIX = _debug
else ifeq ($(BUILD_MODE), test)
  CXXFLAGS += $(TEST_FLAGS)
  LDFLAGS_COMMON += -fsanitize=address,undefined
  BIN_SUFFIX = _test
else
  CXXFLAGS += $(RELEASE_FLAGS)
  BIN_SUFFIX =
endif
# Optional profiling (default: false)
PROFILE ?= 0
ifeq ($(PROFILE), 1)
  CXXFLAGS += -pg
  LDFLAGS_COMMON += -pg
	BIN_SUFFIX := $(BIN_SUFFIX)_profile
endif

# Store base build dir for clean
BASE_BUILD_DIR := $(BUILD_DIR)
MODE_BUILD_DIR := $(BASE_BUILD_DIR)/$(BUILD_MODE)
# Append -profile for profiling builds
ifeq ($(PROFILE), 1)
	MODE_BUILD_DIR := $(MODE_BUILD_DIR)-profile
endif

# Find source files
SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')
TEST_SOURCES := $(shell find $(TEST_DIR) -name '*.cpp')
BENCHMARK_SOURCES := $(shell find $(BENCHMARK_DIR) -name '*.cpp')

# Generate object fnames
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp, $(MODE_BUILD_DIR)/src/%.o, $(SOURCES))
TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cpp, $(MODE_BUILD_DIR)/tests/%.o, $(TEST_SOURCES))
BENCHMARK_OBJECTS := $(patsubst $(BENCHMARK_DIR)/%.cpp, $(MODE_BUILD_DIR)/benchmarks/%.o, $(BENCHMARK_SOURCES))

# Generate dependencies
DEPENDS := $(OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d) $(BENCHMARK_OBJECTS:.o=.d)

# Output target
TARGET := $(BIN_DIR)/$(basename $(TARGET_LIBRARY))$(BIN_SUFFIX).a
TEST_TARGET := $(BIN_DIR)/$(TEST_EXEC)$(BIN_SUFFIX)
BENCHMARK_TARGET := $(BIN_DIR)/$(BENCHMARK_EXEC)$(BIN_SUFFIX)

# Default rule
all: $(TARGET)

# Archive the native library; linking an executable would require main().
$(TARGET) : $(OBJECTS) | $(BIN_DIR)
	$(AR) rcs $@ $^

# Link and build test executable
$(TEST_TARGET): $(OBJECTS) $(TEST_OBJECTS) | $(BIN_DIR)
	$(CXX) $(LIB_PATHS) $^ -o $@ $(LDFLAGS_COMMON) $(LDFLAGS_TEST)

# Benchmarks share the warning setup and fixtures with the unit tests.
$(BENCHMARK_TARGET): $(OBJECTS) $(BENCHMARK_OBJECTS) $(MODE_BUILD_DIR)/tests/test_main.o | $(BIN_DIR)
	$(CXX) $(LIB_PATHS) $^ -o $@ $(LDFLAGS_COMMON) $(LDFLAGS_TEST)

# Compile source files
$(MODE_BUILD_DIR)/src/%.o: $(SRC_DIR)/%.cpp | $(MODE_BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Compile tests
$(MODE_BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.cpp | $(MODE_BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(INCLUDE_FLAGS_TEST) $(CXXFLAGS) -c $< -o $@

# Compile benchmarks
$(MODE_BUILD_DIR)/benchmarks/%.o: $(BENCHMARK_DIR)/%.cpp | $(MODE_BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(INCLUDE_FLAGS_TEST) $(CXXFLAGS) -c $< -o $@

# Create build dirs
$(MODE_BUILD_DIR):
	mkdir -p $@

$(BIN_DIR):
	mkdir -p $@

-include $(DEPENDS)

# Run tests
test: $(TEST_TARGET)
	@echo "Build complete. Run tests with: $(TEST_TARGET)"

benchmarks: $(BENCHMARK_TARGET)
	@echo "Build complete. Run benchmarks with: $(BENCHMARK_TARGET) '[!benchmark]'"

# Clean up only a specific build-mode/profile dir/exec
clean-mode:
	rm -rf $(MODE_BUILD_DIR) $(TARGET) $(TEST_TARGET) $(BENCHMARK_TARGET)

# Clean up everything
clean:
	rm -rf $(BASE_BUILD_DIR) $(BIN_DIR)

.PHONY: all clean clean-mode test benchmarks

#NOTE: careful to test/checkout eigen opt flags, e.g. -DEIGEN_USE_THREADS -DEIGEN_DONT_ALIGN
