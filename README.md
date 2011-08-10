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


