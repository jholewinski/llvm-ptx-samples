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

#include <iostream>
#include <fstream>
#include <cmath>

#include <sys/time.h>

#include "cuda.h"


typedef float Real;


//==--- Utility Functions --------------------------------------------------== //
const char * statusToString(CUresult error)
{
  switch (error) {
    case CUDA_SUCCESS: return "No errors";
    case CUDA_ERROR_INVALID_VALUE: return "Invalid value";
    case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory";
    case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized";
    case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized";

    case CUDA_ERROR_NO_DEVICE: return "No CUDA-capable device available";
    case CUDA_ERROR_INVALID_DEVICE: return "Invalid device";

    case CUDA_ERROR_INVALID_IMAGE: return "Invalid kernel image";
    case CUDA_ERROR_INVALID_CONTEXT: return "Invalid context";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current";
    case CUDA_ERROR_MAP_FAILED: return "Map failed";
    case CUDA_ERROR_UNMAP_FAILED: return "Unmap failed";
    case CUDA_ERROR_ARRAY_IS_MAPPED: return "Array is mapped";
    case CUDA_ERROR_ALREADY_MAPPED: return "Already mapped";
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No binary for GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED: return "Already acquired";
    case CUDA_ERROR_NOT_MAPPED: return "Not mapped";

    case CUDA_ERROR_INVALID_SOURCE: return "Invalid source";
    case CUDA_ERROR_FILE_NOT_FOUND: return "File not found";

    case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle";

    case CUDA_ERROR_NOT_FOUND: return "Not found";

    case CUDA_ERROR_NOT_READY: return "CUDA not ready";

    case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources";
    case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing";

    case CUDA_ERROR_UNKNOWN: return "Unknown error";
    default: return "Unknown error ID";
  }
}

void checkSuccess(CUresult    status,
                  const char *func,
                  const char *errorBuffer = 0)
{
  if (status != CUDA_SUCCESS) {
    if (errorBuffer != 0) {
      std::cerr << "ERROR LOG:" << std::endl
                << errorBuffer << std::endl;
    }

    std::cerr << "ERROR: Could not execute '" << func << "', error ("
              << status << ") " << statusToString(status) << std::endl;
    exit(1);
  }
}

double getTimeStamp()
{
  struct timezone Tzp;
  struct timeval  Tp;
  int             stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0)
    std::cerr << "Error return from gettimeofday: " << stat << "\n";
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}



//==--- Entry Point --------------------------------------------------------== //

int main(int argc,
         char** argv) {

  CUcontext  context;
  CUdevice   device;
  CUmodule   module;
  CUresult   status;
  CUfunction function;
  
  const int kLogSize = 1024;
  char      logBuffer[kLogSize];

  int blockSizeX        = 16;
  int blockSizeY        = 16;
  int blockSizeMultiple = 50;
  int problemSizeX      = blockSizeX * blockSizeMultiple;
  int problemSizeY      = blockSizeY * blockSizeMultiple;

  // Seed random number generator
  srand(time(NULL));
  
  // Initialize CUDA
  std::cout << "Initializing CUDA\n";
  checkSuccess(cuInit(0), "cuInit");
  std::cout << "Selecting first compute device\n";
  checkSuccess(cuDeviceGet(&device, 0), "cuDeviceGet");
  std::cout << "Creating CUDA context\n";
  checkSuccess(cuCtxCreate(&context, 0, device), "cuCtxCreate");

  // Read the PTX kernel from disk
  std::ifstream kernelFile("matrix-multiply.kernel.ptx");
  if (!kernelFile.is_open()) {
    std::cerr << "Failed to open matrix-multiply.kernel.ptx\n";
    return 1;
  }

  // Load entire kernel into a string
  std::string source(std::istreambuf_iterator<char>(kernelFile),
                     (std::istreambuf_iterator<char>()));

  // Configure JIT options
  CUjit_option jitOptions[] = { CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                CU_JIT_ERROR_LOG_BUFFER };
  void* jitOptionValues[]   = { reinterpret_cast<void*>(kLogSize), logBuffer };

  // Load the kernel onto the device
  status = cuModuleLoadDataEx(&module, source.c_str(),
                              sizeof(jitOptions)/sizeof(jitOptions[0]),
                              jitOptions, jitOptionValues);
  checkSuccess(status, "cuModuleLoadDataEx", logBuffer);

  status = cuModuleGetFunction(&function, module, "matrix_multiply");
  checkSuccess(status, "cuModuleGetFunction");


  // Print some diagnostics about the kernel compilation
  int numRegisters;
  cuFuncGetAttribute(&numRegisters, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
  std::cout << "Register Usage:  " << numRegisters << "\n";
  

  // Setup buffers
  Real* hostA = new Real[problemSizeX * problemSizeY];
  Real* hostB = new Real[problemSizeX * problemSizeY];
  Real* refC  = new Real[problemSizeX * problemSizeY];
  Real* cmpC  = new Real[problemSizeX * problemSizeY];

  std::cout << "Problem Size:  " << problemSizeX << " x " << problemSizeY
            << "\n";
  
  CUdeviceptr deviceA;
  CUdeviceptr deviceB;
  CUdeviceptr deviceC;

  status = cuMemAlloc(&deviceA, problemSizeX * problemSizeY * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");
  status = cuMemAlloc(&deviceB, problemSizeX * problemSizeY * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");
  status = cuMemAlloc(&deviceC, problemSizeX * problemSizeY * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");


  // Populate arrays with test data
  for (int i = 0; i < problemSizeX*problemSizeY; ++i) {
    hostA[i] = hostB[i] = (Real)rand() / ((Real)RAND_MAX + (Real)1.0);
    refC[i]  = cmpC[i] = (Real)0.0;
  }


  // Copy buffers to device
  status = cuMemcpyHtoD(deviceA, hostA, problemSizeX * problemSizeY * sizeof(Real));
  checkSuccess(status, "cuMemcpyHtoD");
  status = cuMemcpyHtoD(deviceB, hostB, problemSizeX * problemSizeY * sizeof(Real));
  checkSuccess(status, "cuMemcpyHtoD");
  status = cuMemcpyHtoD(deviceC, cmpC, problemSizeX * problemSizeY * sizeof(Real));
  checkSuccess(status, "cuMemcpyHtoD");

  // Setup block shape
  status = cuFuncSetBlockShape(function, blockSizeX, blockSizeY, 1);
  checkSuccess(status, "cuFuncSetBlockShape");


  // Bind kernel paramters
  status = cuParamSetv(function, 0, &deviceA, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");
  status = cuParamSetv(function, sizeof(CUdeviceptr), &deviceB, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");
  status = cuParamSetv(function, 2*sizeof(CUdeviceptr), &deviceC, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");

  status = cuParamSetSize(function, 3*sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetSize");


  // Launch the kernel
  double deviceStart = getTimeStamp();
  
  status = cuLaunchGrid(function, blockSizeMultiple, blockSizeMultiple);
  checkSuccess(status, "cuLaunchGrid");
  cuCtxSynchronize();

  double deviceEnd = getTimeStamp();
  

  // Copy results back to the host
  status = cuMemcpyDtoH(cmpC, deviceC, problemSizeX * problemSizeY * sizeof(Real));
  checkSuccess(status, "cuMemoryDtoH");


  // Compute the reference solution
  double hostStart = getTimeStamp();

  for (int k = 0; k < problemSizeX; ++k) {
    for (int i = 0; i < problemSizeY; ++i) {
      for (int j = 0; j < problemSizeX; ++j) {
        refC[i*problemSizeX+j] += hostA[i*problemSizeX+k]
          * hostB[k*problemSizeX+j];
      }
    }
  }

  double hostEnd = getTimeStamp();


  // Compare the results
  Real errorNorm = (Real)0.0;
  Real refNorm   = (Real)0.0;
  
  for (int i = 0; i < problemSizeX*problemSizeY; ++i) {
    
    Real diff = refC[i] - cmpC[i];

    errorNorm += diff * diff;
    
    refNorm += refC[i] * refC[i];
  }

  errorNorm = (Real)sqrt((double)errorNorm);
  refNorm   = (Real)sqrt((double)refNorm);

  if ((errorNorm / refNorm) < (Real)1e-5) {
    std::cout << "Host reference comparison test PASSED\n";
  }
  else {
    std::cout << "Host reference comparison test FAILED\n";
    std::cout << "Error Norm:  " << errorNorm << "\n";
    std::cout << "Ref Norm:    " << refNorm << "\n";
  }

#if defined(DEBUG)
  for (int i = 0; i < problemSizeY; ++i) {
    for (int j = 0; j < problemSizeX; ++j) {
      std::cout << refC[i*problemSizeX+j] << ", " << cmpC[i*problemSizeX+j] << "\n";
    }
  }
#endif

  double gflopsDevice = (double)(problemSizeX * problemSizeY * problemSizeX * 2) / (deviceEnd - deviceStart) / 1e9;
  double gflopsHost   = (double)(problemSizeX * problemSizeY * problemSizeX * 2) / (hostEnd - hostStart) / 1e9;
  
  std::cout << "Device Time:     " << (deviceEnd - deviceStart) << "s\n";
  std::cout << "Device GFlop/s:  " << gflopsDevice << "\n";
  std::cout << "Host Time:       " << (hostEnd - hostStart) << "s\n";
  std::cout << "Host GFlop/s:    " << gflopsHost << "\n";
  
  return 0;
}

