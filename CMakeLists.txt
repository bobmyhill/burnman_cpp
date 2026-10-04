cmake_minimum_required(VERSION 3.18)
project(burnman_cpp VERSION 0.1.0 LANGUAGES CXX)

option(BURNMAN_BUILD_PYTHON "Build the Python extension" ON)
option(BURNMAN_BUILD_TESTS "Build the existing C++ tests" OFF)

# Let CMake find standard Homebrew installations as well as system packages.
if(APPLE)
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
if(Eigen3_VERSION VERSION_LESS 3.4)
  message(FATAL_ERROR "BurnMan requires Eigen >= 3.4 (found ${Eigen3_VERSION})")
endif()
find_package(GSL REQUIRED)
find_path(CDD_INCLUDE_DIR NAMES cdd.h PATH_SUFFIXES cddlib cdd REQUIRED)
find_library(CDD_GMP_LIBRARY NAMES cddgmp REQUIRED)
find_path(GMP_INCLUDE_DIR NAMES gmp.h REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)

file(GLOB_RECURSE BURNMAN_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(burnman_core STATIC ${BURNMAN_SOURCES})
target_compile_features(burnman_core PUBLIC cxx_std_17)
target_include_directories(burnman_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(burnman_core PUBLIC Eigen3::Eigen GSL::gsl GSL::gslcblas)
target_include_directories(burnman_core PRIVATE ${CDD_INCLUDE_DIR} ${GMP_INCLUDE_DIR})
target_link_libraries(burnman_core PUBLIC ${CDD_GMP_LIBRARY} ${GMP_LIBRARY})
set_target_properties(burnman_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BURNMAN_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 3.0 CONFIG REQUIRED)
  pybind11_add_module(_core
    python/bindings/module.cpp
    python/bindings/params.cpp
    python/bindings/materials.cpp
    python/bindings/composition.cpp
    python/bindings/combined.cpp
    python/bindings/modifiers.cpp
    python/bindings/solutions.cpp
    python/bindings/minerals.cpp
    python/bindings/polytope.cpp
    python/bindings/equilibration.cpp)
  target_link_libraries(_core PRIVATE burnman_core)
  target_compile_definitions(_core PRIVATE BURNMAN_VERSION="${PROJECT_VERSION}")
  install(TARGETS _core LIBRARY DESTINATION burnman_cpp RUNTIME DESTINATION burnman_cpp)
endif()

if(BURNMAN_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3.4 REQUIRED)
  file(GLOB_RECURSE BURNMAN_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
  add_executable(burnman_tests ${BURNMAN_TEST_SOURCES})
  target_include_directories(burnman_tests PRIVATE tests/include)
  target_link_libraries(burnman_tests PRIVATE burnman_core Catch2::Catch2WithMain)
  add_test(NAME burnman_tests COMMAND burnman_tests "~[benchmark]")
endif()
