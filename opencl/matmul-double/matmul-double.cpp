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
#include <fstream>
#include "common/OCLSample.hpp"

#define BLOCK_SIZE 16

class MatMulDoubleSample : public OCLSample {
public:

  MatMulDoubleSample();

protected:

  virtual void initialize();
  virtual void createMemoryBuffers();
  virtual void setupKernel(cl::Kernel kernel);
  virtual void finishKernel(cl::Kernel kernel);
  virtual void runKernel(cl::Kernel kernel, cl::Event* evt);

private:

  cl::Program programCL_;
  cl::Program programPTX_;

  cl::Buffer  deviceA_;
  cl::Buffer  deviceB_;
  cl::Buffer  deviceC_;

  double*     hostA_;
  double*     hostB_;
  double*     hostC_;

  unsigned int ProblemSize_;
  unsigned int ArraySize_;
};


MatMulDoubleSample::MatMulDoubleSample() {
  ProblemSize_ = 4096;
  ArraySize_ = ProblemSize_ * ProblemSize_;
}

void MatMulDoubleSample::initialize() {
  cl_int result;

  programCL_ = compileSource("matmul-double_kernel.cl");
  programPTX_ = loadBinary("matmul-double_kernel.ptx");

  cl::Kernel kernelCL = cl::Kernel(programCL_, "matmul", &result);
  assert(result == CL_SUCCESS && "Failed to extract kernel");
  cl::Kernel kernelPTX = cl::Kernel(programPTX_, "matmul", &result);
  assert(result == CL_SUCCESS && "Failed to extract kernel");

  setSourceKernel(kernelCL);
  setBinaryKernel(kernelPTX);

  // For this sample, let's run 16 iterations
  setNumberOfIterations(16);
}

void MatMulDoubleSample::runKernel(cl::Kernel kernel, cl::Event* evt) {
  cl_int result;
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
                                                  evt);
  assert(result == CL_SUCCESS && "Failed to launch kernel");
}

void MatMulDoubleSample::createMemoryBuffers() {
  cl_int result;

  // Create host buffers
  hostA_ = new double[ArraySize_];
  hostB_ = new double[ArraySize_];
  hostC_ = new double[ArraySize_];

  // Create device buffers
  deviceA_ = cl::Buffer(getContext(), CL_MEM_READ_ONLY,
                        ArraySize_*sizeof(double), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
  deviceB_ = cl::Buffer(getContext(), CL_MEM_READ_ONLY,
                        ArraySize_*sizeof(double), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
  deviceC_ = cl::Buffer(getContext(), CL_MEM_WRITE_ONLY,
                        ArraySize_*sizeof(double), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
}

void MatMulDoubleSample::setupKernel(cl::Kernel kernel) {
  cl_int result;

  // Copy data to device
  result = getCommandQueue().enqueueWriteBuffer(deviceA_, CL_TRUE, 0,
                                                ArraySize_*sizeof(double),
                                                hostA_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to device");
  result = getCommandQueue().enqueueWriteBuffer(deviceB_, CL_TRUE, 0,
                                                ArraySize_*sizeof(double),
                                                hostB_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to device");
}

void MatMulDoubleSample::finishKernel(cl::Kernel kernel) {
  cl_int result;

    // Copy data back to host
  result = getCommandQueue().enqueueReadBuffer(deviceC_, CL_TRUE, 0,
                                                ArraySize_*sizeof(double),
                                                hostC_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to host");
}

int main(int argc, char** argv) {
  MatMulDoubleSample sample;

  sample.run();

  return 0;
}
