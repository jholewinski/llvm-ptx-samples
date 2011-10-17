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

class Blur2DSample : public OCLSample {
public:

  Blur2DSample();

protected:

  virtual void initialize();
  virtual void createMemoryBuffers();
  virtual void setupKernel(cl::Kernel kernel);
  virtual void finishKernel(cl::Kernel kernel);
  virtual void runKernel(cl::Kernel kernel, cl::Event* evt);

private:

  cl::Program programCL_;
  cl::Program programPTX_;

  cl::Buffer  deviceIn_;
  cl::Buffer  deviceOut_;

  float*      hostIn_;
  float*      hostOut_;

  unsigned int ProblemSize_;
  unsigned int ArraySize_;
};


Blur2DSample::Blur2DSample() {
  ProblemSize_ = 4096;
  ArraySize_ = (ProblemSize_+2) * (ProblemSize_+2);
}

void Blur2DSample::initialize() {
  cl_int result;

  programCL_ = compileSource("blur2d_kernel.cl");
  programPTX_ = loadBinary("blur2d_kernel.ptx");

  cl::Kernel kernelCL = cl::Kernel(programCL_, "blur2d", &result);
  assert(result == CL_SUCCESS && "Failed to extract kernel");
  cl::Kernel kernelPTX = cl::Kernel(programPTX_, "blur2d", &result);
  assert(result == CL_SUCCESS && "Failed to extract kernel");

  setSourceKernel(kernelCL);
  setBinaryKernel(kernelPTX);

  // For this sample, let's run 16 iterations
  setNumberOfIterations(16);
}

void Blur2DSample::runKernel(cl::Kernel kernel, cl::Event* evt) {
  cl_int result;
  cl::NDRange globalSize(ProblemSize_, ProblemSize_);
  cl::NDRange localSize(BLOCK_SIZE, BLOCK_SIZE);

  result = kernel.setArg(0, deviceIn_);
  assert(result == CL_SUCCESS && "Failed to set kernel argument 0");
  result = kernel.setArg(1, deviceOut_);
  assert(result == CL_SUCCESS && "Failed to set kernel argument 1");
  result = kernel.setArg(2, ProblemSize_);
  assert(result == CL_SUCCESS && "Failed to set kernel argument 2");

  result = getCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                                  globalSize, localSize, 0,
                                                  evt);
  assert(result == CL_SUCCESS && "Failed to launch kernel");
}

void Blur2DSample::createMemoryBuffers() {
  cl_int result;

  // Create host buffers
  hostIn_ = new float[ArraySize_];
  hostOut_ = new float[ArraySize_];

  // Create device buffers
  deviceIn_ = cl::Buffer(getContext(), CL_MEM_READ_ONLY,
                         ArraySize_*sizeof(float), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
  deviceOut_ = cl::Buffer(getContext(), CL_MEM_WRITE_ONLY,
                          ArraySize_*sizeof(float), NULL, &result);
  assert(result == CL_SUCCESS && "Failed to allocate device buffer");
}

void Blur2DSample::setupKernel(cl::Kernel kernel) {
  cl_int result;

  // Copy data to device
  result = getCommandQueue().enqueueWriteBuffer(deviceIn_, CL_TRUE, 0,
                                                ArraySize_*sizeof(float),
                                                hostIn_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to device");
}

void Blur2DSample::finishKernel(cl::Kernel kernel) {
  cl_int result;

    // Copy data back to host
  result = getCommandQueue().enqueueReadBuffer(deviceOut_, CL_TRUE, 0,
                                                ArraySize_*sizeof(float),
                                                hostOut_, NULL, NULL);
  assert(result == CL_SUCCESS && "Failed to queue data copy to host");
}

int main(int argc, char** argv) {
  Blur2DSample sample;

  sample.run();

  return 0;
}
