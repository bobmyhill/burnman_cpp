cmake_minimum_required(VERSION 3.18)

# Match native dependency builds unless the caller selects a deployment target.
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND
   "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
  execute_process(COMMAND sw_vers -productVersion
                  OUTPUT_VARIABLE BURNMAN_HOST_MACOS_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "${BURNMAN_HOST_MACOS_VERSION}"
      CACHE STRING "Minimum supported macOS version")
endif()
# Keep the installed package metadata and native __version__ in agreement.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" BURNMAN_VERSION_LINE
     REGEX "^version = ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" BURNMAN_VERSION "${BURNMAN_VERSION_LINE}")
project(burnman_cpp VERSION ${BURNMAN_VERSION} LANGUAGES CXX)

option(BURNMAN_BUILD_PYTHON "Build the Python extension" ON)
option(BURNMAN_BUILD_TESTS "Build the C++ unit tests" OFF)
option(BURNMAN_BUILD_BENCHMARKS "Build the C++ performance benchmarks" OFF)
option(BURNMAN_WARNINGS_AS_ERRORS "Treat project compiler warnings as errors" OFF)

function(burnman_compiler_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
    if(BURNMAN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(BURNMAN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()
endfunction()

# Let CMake find standard Homebrew installations as well as system packages.
set(BURNMAN_WHEEL_DEPENDENCIES "" CACHE PATH "Private prefix for wheel dependencies")
if(BURNMAN_WHEEL_DEPENDENCIES)
  list(PREPEND CMAKE_PREFIX_PATH "${BURNMAN_WHEEL_DEPENDENCIES}")
  set(GSL_ROOT_DIR "${BURNMAN_WHEEL_DEPENDENCIES}")
endif()
if(APPLE)
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
if(Eigen3_VERSION VERSION_LESS 3.4)
  message(FATAL_ERROR "BurnMan requires Eigen >= 3.4 (found ${Eigen3_VERSION})")
endif()
find_package(GSL REQUIRED)
find_path(NLOPT_INCLUDE_DIR NAMES nlopt.hpp REQUIRED)
find_library(NLOPT_LIBRARY NAMES nlopt REQUIRED)
find_path(CDD_INCLUDE_DIR NAMES cdd.h PATH_SUFFIXES cddlib cdd REQUIRED)
find_library(CDD_GMP_LIBRARY NAMES cddgmp REQUIRED)
find_path(GMP_INCLUDE_DIR NAMES gmp.h REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)

file(GLOB_RECURSE BURNMAN_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(burnman_core STATIC ${BURNMAN_SOURCES})
target_compile_features(burnman_core PUBLIC cxx_std_17)
target_include_directories(burnman_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(burnman_core PUBLIC Eigen3::Eigen GSL::gsl GSL::gslcblas)
target_include_directories(burnman_core SYSTEM PRIVATE ${CDD_INCLUDE_DIR} ${GMP_INCLUDE_DIR} ${NLOPT_INCLUDE_DIR})
target_link_libraries(burnman_core PUBLIC ${CDD_GMP_LIBRARY} ${GMP_LIBRARY} ${NLOPT_LIBRARY})
set_target_properties(burnman_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
burnman_compiler_warnings(burnman_core)

if(BURNMAN_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 3.0 CONFIG REQUIRED)
  pybind11_add_module(_core
    python/bindings/module.cpp
    python/bindings/params.cpp
    python/bindings/eos.cpp
    python/bindings/materials.cpp
    python/bindings/composition.cpp
    python/bindings/combined.cpp
    python/bindings/modifiers.cpp
    python/bindings/solutions.cpp
    python/bindings/minerals.cpp
    python/bindings/utils.cpp
    python/bindings/model_sets.cpp
    python/bindings/polytope.cpp
    python/bindings/phase_diagram.cpp
    python/bindings/equilibration.cpp)
  target_link_libraries(_core PRIVATE burnman_core)
  target_compile_definitions(_core PRIVATE BURNMAN_VERSION="${PROJECT_VERSION}")
  burnman_compiler_warnings(_core)
  install(TARGETS _core LIBRARY DESTINATION burnman_cpp RUNTIME DESTINATION burnman_cpp)
  if(BURNMAN_WHEEL_DEPENDENCIES)
    # Let the wheel repair tool find libraries that use relative install names.
    set_target_properties(_core PROPERTIES
                          INSTALL_RPATH "${BURNMAN_WHEEL_DEPENDENCIES}/lib")
    install(DIRECTORY "${BURNMAN_WHEEL_DEPENDENCIES}/share/licenses/"
            DESTINATION burnman_cpp/native_licenses)
  endif()
endif()

if(BURNMAN_BUILD_TESTS OR BURNMAN_BUILD_BENCHMARKS)
  enable_testing()
  find_package(Catch2 3.4 REQUIRED)
endif()

if(BURNMAN_BUILD_TESTS)
  file(GLOB_RECURSE BURNMAN_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
  add_executable(burnman_tests ${BURNMAN_TEST_SOURCES})
  target_include_directories(burnman_tests PRIVATE tests/include)
  target_link_libraries(burnman_tests PRIVATE burnman_core Catch2::Catch2WithMain)
  burnman_compiler_warnings(burnman_tests)
  add_test(NAME burnman_tests COMMAND burnman_tests)
endif()

if(BURNMAN_BUILD_BENCHMARKS)
  file(GLOB_RECURSE BURNMAN_BENCHMARK_SOURCES CONFIGURE_DEPENDS benchmarks/*.cpp)
  add_executable(burnman_benchmarks ${BURNMAN_BENCHMARK_SOURCES} tests/test_main.cpp)
  target_include_directories(burnman_benchmarks PRIVATE tests/include)
  target_link_libraries(burnman_benchmarks PRIVATE burnman_core Catch2::Catch2WithMain)
  burnman_compiler_warnings(burnman_benchmarks)
  add_test(NAME burnman_benchmarks COMMAND burnman_benchmarks "[!benchmark]"
    --benchmark-samples 20 --benchmark-resamples 1000 --benchmark-warmup-time 10
    --reporter "xml::out=${CMAKE_CURRENT_BINARY_DIR}/benchmark-report.xml"
    --reporter console)
  set_tests_properties(burnman_benchmarks PROPERTIES LABELS benchmarks TIMEOUT 300)
endif()
