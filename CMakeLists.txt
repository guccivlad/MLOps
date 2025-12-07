cmake_minimum_required(VERSION 3.15)
project(mygauss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(mygauss src/main.cpp)

install(TARGETS mygauss DESTINATION .)