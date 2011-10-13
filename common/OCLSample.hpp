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

  virtual void initialize();
  virtual void runSourceKernel();
  virtual void runBinaryKernel();

  cl::Program compileSource(const std::string& source);
  cl::Program loadBinary(const std::string& binary);


  cl::Context& getContext() {
    return context_;
  }

  cl::CommandQueue& getCommandQueue() {
    return queue_;
  }

private:

  void initOpenCL();

  cl::Platform     platform_;
  cl::Device       device_;
  cl::Context      context_;
  cl::CommandQueue queue_;
};

#endif
