cmake_minimum_required(VERSION 3.10)
project(moveo_driver VERSION 1.3.0)

# Compile flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-stringop-overflow -O3 -DNDEBUG")

# Find pybind11
find_package(pybind11 REQUIRED)

# Add include directories
include_directories(
    ${PROJECT_BINARY_DIR}
    include
)

# Define the Python module
pybind11_add_module(moveo_driver
    src/moveo_driver.cpp
    src/moveo.cpp
    src/serial_cobs.cpp
)
