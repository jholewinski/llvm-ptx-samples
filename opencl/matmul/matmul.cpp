/*
 * Copyright (C) 2011 by Justin Holewinski
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cassert>
#include <iostream>
#include "common/OCLSample.hpp"

#define BLOCK_SIZE 16

char CLKernel[] = {
  #include "matmul_kernel.cl.h"
  , 0x0
};

char PTXKernel[] = {
  #include "matmul_kernel.ptx.h"
  , 0x0
};

class MatMulSample : public OCLSample {
public:

  MatMulSample();

protected:

  virtual void initialize();
  virtual void runSourceKernel();
  virtual void runBinaryKernel();

private:

  void runKernel(cl::Kernel& kernel);

  cl::Program programCL_;
  cl::Kernel  kernelCL_;
  cl::Program programPTX_;
  cl::Kernel  kernelPTX_;

  cl::Buffer  deviceA_;
  cl::Buffer  deviceB_;
  cl::Buffer  deviceC_;

  float*      hostA_;
  float*      hostB_;
  float*      hostC_;

  unsigned int ProblemSize_;
  unsigned int ArraySize_;
};


MatMulSample::MatMulSample() {
  ProblemSize_ = 4096;
  ArraySize_ = ProblemSize_ * ProblemSize_;
}

void MatMulSample::initialize() {
  cl_int result;

  std::cout << "Using problem size: " << ProblemSize_ << " x " << ProblemSize_
            << "\n";

  // Create host buffers
  hostA_ = new float[ArraySize_];
  hostB_ = new float[ArraySize_];
  hostC_ = new float[ArraySize_];

  programCL_  = compileSource(CLKernel);
  programPTX_ = loadBinary(PTXKernel);

  kernelCL_  = cl::Kernel(programCL_, "matmul", &result);
  assert(result == CL_SUCCESS && "Failed to extract kernel");
  kernelPTX_ = cl::Kernel(programPTX_, "matmul", &result);
  assert(result == CL_SUCCESS && "Failed to extract kernel");

  // Create device memory buffers
  deviceA_ = cl::Buffer(getContext(), CL_MEM_READ_ONLY,
                        ArraySize_*sizeof(float), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
  deviceB_ = cl::Buffer(getContext(), CL_MEM_READ_ONLY,
                        ArraySize_*sizeof(float), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
  deviceC_ = cl::Buffer(getContext(), CL_MEM_WRITE_ONLY,
                        ArraySize_*sizeof(float), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
}

void MatMulSample::runKernel(cl::Kernel& kernel) {
  cl_int result;
  cl::Event kernelEvent;
  cl_ulong kernelStart, kernelEnd;

  // Copy data to device
  result = getCommandQueue().enqueueWriteBuffer(deviceA_, CL_TRUE, 0,
                                                ArraySize_*sizeof(float),
                                                hostA_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to device");
  result = getCommandQueue().enqueueWriteBuffer(deviceB_, CL_TRUE, 0,
                                                ArraySize_*sizeof(float),
                                                hostB_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to device");

  cl::NDRange globalSize(ProblemSize_, ProblemSize_);
  cl::NDRange localSize(BLOCK_SIZE, BLOCK_SIZE);

  result = kernel.setArg(0, deviceA_);
  assert(result == CL_SUCCESS && "Failed to set kernel argument 0");
  result = kernel.setArg(1, deviceB_);
  assert(result == CL_SUCCESS && "Failed to set kernel argument 1");
  result = kernel.setArg(2, deviceC_);
  assert(result == CL_SUCCESS && "Failed to set kernel argument 2");

  result = getCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                                  globalSize, localSize, 0,
                                                  &kernelEvent);
  assert(result == CL_SUCCESS && "Failed to launch kernel");

  getCommandQueue().flush();
  kernelEvent.wait();

  // Copy data back to host
  result = getCommandQueue().enqueueReadBuffer(deviceC_, CL_TRUE, 0,
                                                ArraySize_*sizeof(float),
                                                hostC_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to host");

  result = kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START,
                                                  &kernelStart);
  assert(result == CL_SUCCESS && "Unable to get profiling information");
  result = kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,
                                                  &kernelEnd);
  assert(result == CL_SUCCESS && "Unable to get profiling information");

  double elapsed = (double)1e-9 * (kernelEnd - kernelStart);
  std::cout << "Elapsed: " << elapsed << "s\n";
}

void MatMulSample::runSourceKernel() {
  runKernel(kernelCL_);
}

void MatMulSample::runBinaryKernel() {
  runKernel(kernelPTX_);
}

int main(int argc, char** argv) {
  MatMulSample sample;

  sample.run();

  return 0;
}
