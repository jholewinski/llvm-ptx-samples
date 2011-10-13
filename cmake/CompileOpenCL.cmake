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

set(LIBCLC ${CMAKE_SOURCE_DIR}/libclc)
set(CLANG_OPENCL_CFLAGS -ccc-host-triple ptx32 -S -emit-llvm -I${LIBCLC}/include/ptx -I${LIBCLC}/include/generic -include clc/clc.h -Dcl_clang_storage_class_specifiers)

macro(compile_opencl _ptxout _src)
  message(STATUS "OpenCL: ${CMAKE_CURRENT_SOURCE_DIR}/${_src} -> ${CMAKE_CURRENT_BINARY_DIR}/${_src}.ll")
  add_custom_command(OUTPUT  ${_src}.ll
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
                     COMMAND ${CLANG_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/${_src} ${CLANG_OPENCL_CFLAGS} -o ${_src}.ll
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_command(OUTPUT  ${_src}.opt.ll
                     DEPENDS ${_src}.ll
                     COMMAND ${OPT_PROGRAM} -O3 -loop-unroll -S ${_src}.ll -o ${_src}.opt.ll
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_command(OUTPUT  ${_ptxout}
                     DEPENDS ${_src}.opt.ll
                     COMMAND ${LLC_PROGRAM} -march=ptx32 -mattr=ptx23 ${_src}.ll -o ${_ptxout}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${_src}.ll PROPERTIES GENERATED TRUE)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${_src}.opt.ll PROPERTIES GENERATED TRUE)
  set_source_files_properties(${_ptxout} PROPERTIES GENERATED TRUE)
  add_custom_target(${_ptxout}-build DEPENDS ${_ptxout})
endmacro()