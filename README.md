LLVM PTX Samples
================

This collection of sample programs highlights the PTX code generation back-end
for the LLVM project.  These programs serve as both examples for the usage of
the back-end (as well as the Clang front-end integration) and a simple test
suite.

These samples are currently being converted to OpenCL.


Usage
-----

To compile the samples, CMake and the NVidia CUDA toolkit is required, as well as a reasonably up-to-date build of Clang/LLVM which was built with the PTX back-end. It is recommended that you build the examples in a separate directory.

For most systems, the following build commands can be used:

    $ cd llvm-ptx-samples
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

If CMake complains that it cannot find a CUDA toolkit, please add -DCUDA_TOOLKIT_ROOT_DIR=... to your CMake invocation, specifying the root of your CUDA installation. Similarly, if Clang or the other LLVM tools cannot be found, please adjust your PATH to include them.

Once the samples are built, they can be executed from their build directory, which mirrors the layout of the source directory. For example, to run the matrix multiplication sample, you can execute:

    $ ./opencl/matmul/ocl-matmul

from your build directory.
