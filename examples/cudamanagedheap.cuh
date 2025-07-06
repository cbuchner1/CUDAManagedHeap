/*
  ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Rene Widera - r.widera ( at ) hzdr.de

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

#pragma once

#include "tools/managedheap.cuh"
#ifndef HEAPARGS
typedef GPUTools::ManagedHeap<> heap_t;
#else
typedef GPUTools::ManagedHeap<HEAPARGS> heap_t;
#endif

__device__ heap_t* theHeap_pd; // device pointer to the heap
heap_t* theHeap_ph;            // host pointer to the heap

void* initHeap(size_t memsize = 8*1024U*1024U)
{
  void* heap_p;
  theHeap_ph = new heap_t(memsize);
  CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&heap_p, theHeap_pd));
  cudaMemcpy(heap_p, &theHeap_ph, sizeof(heap_t*), cudaMemcpyHostToDevice);
  return heap_p;
}

void destroyHeap()
{
  void* heap_p;
  delete theHeap_ph;
  theHeap_pd = nullptr;
  CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&heap_p, theHeap_pd));
  cudaMemcpy(heap_p, &theHeap_ph, sizeof(heap_t*), cudaMemcpyHostToDevice);
}

#ifdef __CUDACC__
__device__ void* malloc(size_t t) __THROW
{
  return theHeap_pd->alloc(t);
}
__device__ void  free(void* p) __THROW
{
  theHeap_pd->dealloc(p);
}
#endif //__CUDACC__

// explicitly force instantiation of these kernels.
// for whatever reason this is not happening automatically.
static void dummy()
{
  GPUTools::heapAllocKernel<<<1,1>>>(theHeap_ph, 123, nullptr);
  GPUTools::heapDeallocKernel<<<1,1>>>(theHeap_ph, nullptr);
}
