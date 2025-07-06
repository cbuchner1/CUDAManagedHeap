/*
  ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include <cuda.h>
typedef unsigned int uint;

//set the template arguments using HEAPARGS
// pagesize ... byter per page
// accessblocks ... number of superblocks
// regionsize ... number of regions for meta data structure
// wastefactor ... how much memory can be wasted per alloc (multiplicative factor)
// use_coalescing ... combine memory requests within each warp
// resetfreedpages ... allow pages to be reused with a different size
#define HEAPARGS 65536, 8, 16, 2, true, true

#define _DEBUG 1 // makes CUDA_CHECKED_CALL throw

//include the scatter alloc heap
#include "cudamanagedheap.cuh"

#include <iostream>
#include <stdio.h>

void runexample(int cuda_device);


int main(int argc, char** argv)
{
   try
  {
    int cuda_device = argc > 1 ? atoi(argv[1]) : 0;

    cudaDeviceProp deviceProp;
	  CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, cuda_device));
    std::cout << "Using device: " << deviceProp.name << std::endl;

	  if( deviceProp.major < 2 ) {
		  std::cerr << "This GPU with Compute Capability " << deviceProp.major 
        << "." << deviceProp.minor <<  " does not meet minimum requirements." << std::endl;
		  std::cerr << "A GPU with Compute Capability >= 2.0 is required." << std::endl;
      return -2;
	  }
  
    runexample(cuda_device);

    cudaDeviceReset();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what()  << std::endl;
    #ifdef WIN32
    while (!_kbhit());
    #endif
    return -1;
  }
  catch (...)
  {
    std::cout << "unknown exception!" << std::endl;
    #ifdef WIN32
    while (!_kbhit());
    #endif
    return -1;
  }

  return 0;
}

__global__ void allocSomething(uint** parray)
{
  parray[threadIdx.x + blockIdx.x*blockDim.x] = new uint[threadIdx.x % 4];
}
__global__ void freeSomething(uint** parray)
{
  delete[] parray[threadIdx.x + blockIdx.x*blockDim.x];
}


void runexample(int cuda_device)
{
  CUDA_CHECKED_CALL(cudaSetDevice(cuda_device));

  //init the heap
  initHeap();
  //you can also specify the size of the heap in bytes
  //initHeap(8U*1024U*1024U);

  size_t block = 128;
  size_t grid = 64;

  uint** data;
  CUDA_CHECKED_CALL(cudaMallocManaged(&data, grid*block*sizeof(uint*)));

  allocSomething<<<grid,block>>>(data);
  CUDA_CHECKED_CALL(cudaDeviceSynchronize());

  freeSomething<<<grid,block>>>(data);
  CUDA_CHECKED_CALL(cudaDeviceSynchronize());
}
