cmake_minimum_required(VERSION 3.15)
project(CVMLTakehomeChallenge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Download nlohmann/json using FetchContent
include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Download Boost using FetchContent (for Boost Geometry)
# Boost Geometry is header-only, so we download the full Boost headers
# Using Boost 1.83.0 from official source (more stable URL)
FetchContent_Declare(
  boost
  URL https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.gz/download
)
FetchContent_MakeAvailable(boost)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Boost include directory (from FetchContent)
# Boost headers are in the root of the extracted archive
# FetchContent makes the source directory available after MakeAvailable
set(BOOST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/_deps/boost-src")
if(EXISTS "${BOOST_ROOT}/boost/geometry.hpp")
    target_include_directories(${PROJECT_NAME} PRIVATE ${BOOST_ROOT})
    message(STATUS "Boost Geometry found at: ${BOOST_ROOT}")
else()
    message(WARNING "Boost Geometry headers not found at expected location")
endif()

# Define data directory path as compile definition
# This makes the path portable across different build configurations
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}" ABSOLUTE)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    PROJECT_ROOT="${PROJECT_ROOT}"
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# Compiler options
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
