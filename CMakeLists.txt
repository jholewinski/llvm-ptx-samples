#
# Copyright (C) 2011 by Justin Holewinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

cmake_minimum_required(VERSION 2.6)

project(llvm-ptx-samples)

# Locate needed packages
# We use the CUDA package to locate the NVidia SDK for us
find_package(CUDA REQUIRED)

# Locate Clang
find_program(CLANG_PROGRAM NAMES clang)
if(NOT CLANG_PROGRAM)
  message(FATAL_ERROR "Unable to locate Clang.")
else()
  message(STATUS "Found Clang: ${CLANG_PROGRAM}")
endif()

# Locate opt
find_program(OPT_PROGRAM NAMES opt)
if(NOT OPT_PROGRAM)
  message(FATAL_ERROR "Unable to locate opt.")
else()
  message(STATUS "Found opt: ${OPT_PROGRAM}")
endif()

# Locate llc
find_program(LLC_PROGRAM NAMES llc)
if(NOT LLC_PROGRAM)
  message(FATAL_ERROR "Unable to locate llc.")
else()
  message(STATUS "Found opt: ${LLC_PROGRAM}")
endif()

# Locate xxd
find_program(XXD_PROGRAM NAMES xxd)
if(NOT XXD_PROGRAM)
  message(FATAL_ERROR "Unable to locate xxd.")
else()
  message(STATUS "Found xxd: ${XXD_PROGRAM}")
endif()

# Locate OpenCL library
find_library(OPENCL_LIBRARY NAMES OpenCL)
if(NOT OPENCL_LIBRARY)
  message(FATAL_ERROR "Unable to locate OpenCL library.")
else()
  message(STATUS "Found OpenCL: ${OPENCL_LIBRARY}")
endif()

# Include our custom build code
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileOpenCL.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedResource.cmake)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(opencl)
