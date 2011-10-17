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

#if !defined(OCL_SAMPLE_HPP_INC)
#define OCL_SAMPLE_HPP_INC 1

#include "common/cl.hpp"
#include "common/Sample.hpp"
/**
 * Base class for OpenCL samples.
 */
class OCLSample : public Sample {
public:

  OCLSample();

  virtual ~OCLSample();

  virtual void run();

protected:

  /**
   * Hook for samples to perform any initialization.
   */
  virtual void initialize();

  /**
   * Hook for samples to allocate any needed memory buffers.
   */
  virtual void createMemoryBuffers();

  /**
   * Hook for samples to perform any kernel setup, such as copying data from
   * the host to the device.
   */
  virtual void setupKernel(cl::Kernel kernel);

  /**
   * Hook for samples to perform any kernel finalization, such as copying data
   * from the device to the host. */
  virtual void finishKernel(cl::Kernel kernel);

  /**
   * Hook for samples to run the requested kernel.
   */
  virtual void runKernel(cl::Kernel kernel, cl::Event* evt);

  cl::Program compileSource(const std::string& source);
  cl::Program loadBinary(const std::string& binary);

  void setSourceKernel(cl::Kernel kernel) {
    sourceKernel_ = kernel;
  }

  cl::Kernel getSourceKernel() {
    return sourceKernel_;
  }

  void setBinaryKernel(cl::Kernel kernel) {
    binaryKernel_ = kernel;
  }

  cl::Kernel getBinaryKernel() {
    return binaryKernel_;
  }

  cl::Context& getContext() {
    return context_;
  }

  cl::CommandQueue& getCommandQueue() {
    return queue_;
  }

  unsigned getNumberOfIterations() const {
    return numIterations_;
  }

  void setNumberOfIterations(unsigned iters) {
    numIterations_ = iters;
  }

private:

  void initOpenCL();
  void timeKernel(cl::Kernel kernel, double& elapsed, double& average);

  cl::Platform     platform_;
  cl::Device       device_;
  cl::Context      context_;
  cl::CommandQueue queue_;
  cl::Kernel       sourceKernel_;
  cl::Kernel       binaryKernel_;
  unsigned         numIterations_;

};

#endif
