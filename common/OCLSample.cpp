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

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include "common/OCLSample.hpp"

OCLSample::OCLSample()
: numIterations_(4) {
  initOpenCL();
}

OCLSample::~OCLSample() {
}

void OCLSample::initialize() {
}

void OCLSample::createMemoryBuffers() {
}

void OCLSample::setupKernel(cl::Kernel kernel) {
}

void OCLSample::finishKernel(cl::Kernel kernel) {
}

void OCLSample::runKernel(cl::Kernel kernel, cl::Event* evt) {
}

void OCLSample::run() {
  double elapsed, average;

  initialize();
  createMemoryBuffers();

  std::cout << "------------------------------\n";
  std::cout << "* Source Kernel\n";
  std::cout << "------------------------------\n";
  setupKernel(sourceKernel_);
  timeKernel(sourceKernel_, elapsed, average);
  finishKernel(sourceKernel_);

  std::cout << "Number of Iterations: " << numIterations_ << "\n";
  std::cout << "Total Time:           " << elapsed << " sec\n";
  std::cout << "Average Time:         " << average << " sec\n";

  std::cout << "------------------------------\n";
  std::cout << "* Binary Kernel\n";
  std::cout << "------------------------------\n";
  setupKernel(binaryKernel_);
  timeKernel(binaryKernel_, elapsed, average);
  finishKernel(binaryKernel_);

  std::cout << "Number of Iterations: " << numIterations_ << "\n";
  std::cout << "Total Time:           " << elapsed << " sec\n";
  std::cout << "Average Time:         " << average << " sec\n";
}

void OCLSample::timeKernel(cl::Kernel kernel, double& elapsed,
                           double& average) {
  cl::Event* events = new cl::Event[numIterations_];
  cl_int result;

  elapsed = 0.0;

  for(unsigned i = 0; i < numIterations_; ++i) {
    cl_ulong start, end;
    runKernel(kernel, &events[i]);

    queue_.flush();
    events[i].wait();

    result = events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START,
                                                  &start);
    assert(result == CL_SUCCESS && "Unable to get profiling information");
    result = events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,
                                                  &end);
    assert(result == CL_SUCCESS && "Unable to get profiling information");
    elapsed += (double)1e-9 * (end - start);
  }

  average = elapsed / (double)numIterations_;

  delete [] events;
}

cl::Program OCLSample::compileSource(const std::string& filename) {
  cl_int result;

  std::ifstream kernelStream(filename.c_str());
  std::string source(std::istreambuf_iterator<char>(kernelStream),
                     (std::istreambuf_iterator<char>()));
  kernelStream.close();

  cl::Program::Sources sources(1, std::make_pair(source.c_str(),
                                                 source.size()));
  cl::Program program(context_, sources, &result);
  assert(result == CL_SUCCESS && "Failed to load program source");
  std::vector<cl::Device> devices;
  devices.push_back(device_);
  result = program.build(devices);
  if(result != CL_SUCCESS) {
    std::cerr << "Source compilation failed.\n";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
    assert(false && "Unable to continue");
  }
  return program;
}

cl::Program OCLSample::loadBinary(const std::string& filename) {
  cl_int result;

  std::ifstream kernelStream(filename.c_str());
  std::string binary(std::istreambuf_iterator<char>(kernelStream),
                     (std::istreambuf_iterator<char>()));
  kernelStream.close();

  cl::Program::Binaries binaries(1, std::make_pair(binary.c_str(),
                                                   binary.size()));
  std::vector<cl::Device> devices;
  devices.push_back(device_);
  cl::Program program(context_, devices, binaries, NULL, &result);
  assert(result == CL_SUCCESS && "Failed to load program source");
  result = program.build(devices);
  if(result != CL_SUCCESS) {
    std::cerr << "Source compilation failed.\n";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
    assert(false && "Unable to continue");
  }
  return program;
}

void OCLSample::initOpenCL() {
  cl_int result;

  // First, select an OpenCL platform
  typedef std::vector<cl::Platform> PlatformVector;
  PlatformVector platforms;
  result = cl::Platform::get(&platforms);
  assert(result == CL_SUCCESS && "Failed to retrieve OpenCL platform");
  assert(platforms.size() > 0 && "No OpenCL platforms found");

  // For now, just blindly select the first platform.
  // @TODO: Implement a check here for the NVidia OpenCL platform.
  platform_ = platforms[0];

  // Create an OpenCL context.
  cl_context_properties cps[] = { CL_CONTEXT_PLATFORM,
    (cl_context_properties)(platform_)(), 0 };
  context_ = cl::Context(CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &result);
  assert(result == CL_SUCCESS && "Failed to create OpenCL context");

  // Retrieve the available OpenCL devices.
  typedef std::vector<cl::Device> DeviceVector;
  DeviceVector devices;
  devices = context_.getInfo<CL_CONTEXT_DEVICES>();
  assert(devices.size() > 0 && "No OpenCL devices found");

  // For now, just blindly select the first device.
  // @TODO: Implement some sort of device check here.
  device_ = devices[0];

  // Create a command queue
  queue_ = cl::CommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &result);
  assert(result == CL_SUCCESS && "Failed to create command queue");
}
