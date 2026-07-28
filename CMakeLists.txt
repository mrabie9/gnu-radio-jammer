cmake_minimum_required(VERSION 3.16)
project(gr-gpsk_comms VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
endif()

include(CTest)
include(GNUInstallDirs)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Gnuradio "3.10" REQUIRED COMPONENTS runtime blocks fft filter analog digital)
include(GrPython)

add_subdirectory(python)
add_subdirectory(grc)
add_subdirectory(examples)
