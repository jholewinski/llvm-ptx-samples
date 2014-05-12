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

# Compiler flags
set(LIBCLC ${CMAKE_SOURCE_DIR}/libclc)
set(LIBCLC_FLAGS -I${LIBCLC}/include/ptx -I${LIBCLC}/include/generic -include clc/clc.h -Dcl_clang_storage_class_specifiers)
set(CLANG_FLAGS -target nvptx-unknown-nvcl -S -emit-llvm -O4 ${LIBCLC_FLAGS})
set(OPT_FLAGS -O3 -loop-unroll)
set(LLC_FLAGS -mcpu=sm_20)

set(RESOURCE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)

# By default, _llout is assumed to be relative to RESOURCE_OUTPUT_DIR and
# _srcin is assumed to be relative to CMAKE_CURRENT_SOURCE_DIR
macro(compile_opencl_to_llvmir _llout _srcin)
  get_filename_component(_srcin_abs ${_srcin} ABSOLUTE)
  add_custom_command(OUTPUT ${RESOURCE_OUTPUT_DIR}/${_llout}
                     DEPENDS ${_srcin_abs}
                     COMMAND ${CLANG_PROGRAM} ${CLANG_FLAGS} ${_srcin_abs} -o ${_llout}
                     WORKING_DIRECTORY ${RESOURCE_OUTPUT_DIR}
                     COMMENT "Compiling ${_srcin} -> ${_llout}")
  add_custom_target(${_llout} DEPENDS ${RESOURCE_OUTPUT_DIR}/${_llout})
endmacro()

macro(optimize_llvmir _llout _llin)
  set(_llout_abs ${RESOURCE_OUTPUT_DIR}/${_llout})
  set(_llin_abs ${RESOURCE_OUTPUT_DIR}/${_llin})
  add_custom_command(OUTPUT ${_llout_abs}
                     DEPENDS ${_llin_abs}
                     COMMAND ${OPT_PROGRAM} -S ${OPT_FLAGS} ${_llin_abs} -o ${_llout_abs}
                     WORKING_DIRECTORY ${RESOURCE_OUTPUT_DIR}
                     COMMENT "Optimizing ${_llin} -> ${_llout}")
  add_custom_target(${_llout} DEPENDS ${_llout_abs})
endmacro()

macro(codegen_ptx _ptxout _llin)
  set(_ptxout_abs ${RESOURCE_OUTPUT_DIR}/${_ptxout})
  set(_llin_abs ${RESOURCE_OUTPUT_DIR}/${_llin})
  add_custom_command(OUTPUT ${_ptxout_abs}
                     DEPENDS ${_llin_abs}
                     COMMAND ${LLC_PROGRAM} ${LLC_FLAGS} ${_llin_abs} -o ${_ptxout_abs}
                     WORKING_DIRECTORY ${RESOURCE_OUTPUT_DIR}
                     COMMENT "Compiling ${_llin} -> ${_ptxout}")
  add_custom_target(${_ptxout} DEPENDS ${_ptxout_abs})
endmacro()

macro(copy_opencl _clin)
  set(_dest_abs ${RESOURCE_OUTPUT_DIR}/${_clin})
  set(_src_abs ${CMAKE_CURRENT_SOURCE_DIR}/${_clin})
  add_custom_command(OUTPUT ${_dest_abs}
                     DEPENDS ${_src_abs}
                     COMMAND ${CMAKE_COMMAND} -E copy ${_src_abs} ${_dest_abs}
                     WORKING_DIRECTORY ${RESOURCE_OUTPUT_DIR}
                     COMMENT "Copying OpenCL source ${_clin}")
  add_custom_target(${_clin} DEPENDS ${_dest_abs})
endmacro()

macro(create_opencl_targets _targets _kernel)
  set(${_targets})
  compile_opencl_to_llvmir(${_kernel}.ll ${_kernel}.cl)
  optimize_llvmir(${_kernel}.opt.ll ${_kernel}.ll)
  codegen_ptx(${_kernel}.ptx ${_kernel}.opt.ll)
  copy_opencl(${_kernel}.cl)
  list(APPEND ${_targets} ${_kernel}.ptx ${_kernel}.cl)
endmacro()
