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

set(_sources  matmul.cpp)

set(_resources  matmul_kernel.cl.h
                ${CMAKE_CURRENT_BINARY_DIR}/matmul_kernel.ptx.h)

compile_opencl(matmul_kernel.ptx matmul_kernel.cl)

embed_resource(matmul_kernel.cl.h ${CMAKE_CURRENT_SOURCE_DIR}/matmul_kernel.cl)
embed_resource(matmul_kernel.ptx.h matmul_kernel.ptx)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(ocl-matmul ${_sources} ${_resources})
target_link_libraries(ocl-matmul ${OPENCL_LIBRARY} sampleutil)

add_dependencies(ocl-matmul matmul_kernel.cl.h-build matmul_kernel.ptx-build matmul_kernel.ptx.h-build)
