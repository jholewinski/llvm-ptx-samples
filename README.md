LLVM PTX Samples
================

This collection of sample programs highlights the PTX code generation back-end
for the LLVM project.  These programs serve as both examples for the usage of
the back-end (as well as the Clang front-end integration) and a simple test
suite.


Usage
-----

To compile the samples, simple run 'make' from the top-level directory.  On most
*nix distributions, this is sufficient.  The NVidia CUDA Toolkit is assumed to
be installed under /usr/local/cuda, and the Clang compiler is assumed to be in
your PATH environment variable.

The build process can be customized using the following variables:

* CXX - C++ compiler to use (absolute path)
* CUDA_TOOLKIT - Absolute path to CUDA Toolkit installation
* SUBMIT - Command prefix to use for GPU job submission (cluster environments)
* VERBOSE - If set to 1, print command-line invocations

Example:

To use a certain version of GCC as the C++ compiler and a version of the CUDA
Toolkit installed in /opt/cuda, you can build the samples with:

    $ make all CXX=g++-4.5 CUDA_TOOLKIT=/opt/cuda



Examples
--------

### Vector Addition (kernels/vector-add)

This example shows the simplest use-case for generating CUDA kernel code from
Clang.  The implemented kernel performs a simple vector addition, where each
device thread is responsible for computing one element of the output vector.


### Matrix Multiplication (kernels/matrix-multiply)

This examples shows a simple matrix multiplication kernel, where each device
thread is responsible for computing one element of the output matrix.  All
memory accesses are performed on global memory for example purposes.


### Tiled Matrix Multiplication (kernels/matrix-multiply-tiled)

This is a variant of the matrix multiplication example that uses shared memory
to improve the performance of the kernel by processing blocks of the matrix at
a time.  This example shows how to use shared memory in device kernels, as well
as how to use thread synchronization primitives.

