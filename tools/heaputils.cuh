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

#pragma once

#include <string>
#include <sstream>
#include <stdexcept>
#ifndef __CUDA_ARCH__
#include <atomic>
#endif

namespace CUDA
{
  class error : public std::runtime_error
  {
  private:
    static std::string genErrorString(cudaError error, const char* file, int line)
    {
      std::ostringstream msg;
      msg << file << '(' << line << "): error: " << cudaGetErrorString(error);
      return msg.str();
    }
  public:
    error(cudaError error, const char* file, int line)
      : runtime_error(genErrorString(error, file, line))
    {
    }

    error(cudaError error)
      : runtime_error(cudaGetErrorString(error))
    {
    }

    error(const std::string& msg)
      : runtime_error(msg)
    {
    }
  };

  inline void checkError(cudaError error, const char* file, int line)
  {
#ifdef _DEBUG
    if (error != cudaSuccess)
      throw CUDA::error(error, file, line);
#endif
  }

  inline void checkError(const char* file, int line)
  {
    checkError(cudaGetLastError(), file, line);
  }

  inline void checkError()
  {
#ifdef _DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw CUDA::error(error);
#endif
  }

#define CUDA_CHECKED_CALL(call) CUDA::checkError(call, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CUDA::checkError(__FILE__, __LINE__)
}

#ifndef __CUDACC__

// Host fallbacks for CUDA specific functionality
static constexpr int warpSize = 32;

#else

// System Wide Atomics for host and device
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wempty-body"

template<typename T>
inline T __host__ __device__ SWatomicAdd(T* address, T val)
{
#ifdef __CUDA_ARCH__
  return atomicAdd_system(address, val);
#else
  std::atomic_ref<T> ref(*address); return ref.fetch_add(val);
#endif
}
template<typename T>
inline T __host__ __device__ SWatomicSub(T* address, T val)
{
#ifdef __CUDA_ARCH__
  return atomicSub_system(address, val);
#else
  std::atomic_ref<T> ref(*address); return ref.fetch_sub(val);
#endif
}
template<typename T>
inline T __host__ __device__ SWatomicExch(T* address, T val)
{
#ifdef __CUDA_ARCH__
  return atomicExch_system(address, val);
#else
  std::atomic_ref<T> ref(*address); return ref.exchange(val);
#endif
}
template<typename T>
inline T __host__ __device__ SWatomicCAS(T* address, T compare, T val)
{
#ifdef __CUDA_ARCH__
  return atomicCAS_system(address, compare, val);
#else
  std::atomic_ref<T> ref(*address); T expected = compare; ref.compare_exchange_strong(expected, val); return expected;
#endif
}
template<typename T>
inline T __host__ __device__ SWatomicAnd(T* address, T val){
#ifdef __CUDA_ARCH__
  return atomicAnd_system(address, val);
#else
  std::atomic_ref<T> ref(*address); T old = ref.load(); while(!ref.compare_exchange_weak(old, old & val)); return old;
#endif
}
template<typename T>
inline T __host__ __device__ SWatomicOr(T* address, T val){
#ifdef __CUDA_ARCH__
  return atomicOr_system(address, val);
#else
  std::atomic_ref<T> ref(*address); T old = ref.load(); while(!ref.compare_exchange_weak(old, old | val)); return old;
#endif
}
template<typename T>
inline T __host__ __device__ SWatomicMax(T* address, T val){
#ifdef __CUDA_ARCH__
  return atomicMax_system(address, val);
#else
  std::atomic_ref<T> ref(*address); T old = ref.load(); while(old < val && !ref.compare_exchange_weak(old, val)); return old;
#endif
}
template<typename T>
inline T __host__ __device__ SWatomicMin(T* address, T val){
#ifdef __CUDA_ARCH__
  return atomicMin_system(address, val);
#else
  std::atomic_ref<T> ref(*address); T old = ref.load(); while(old > val && !ref.compare_exchange_weak(old, val)); return old;
#endif
}

#pragma clang diagnostic pop
#endif

#ifndef __CUDA_ARCH__
inline void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_block()  { std::atomic_thread_fence(std::memory_order_seq_cst); }

inline int __ffs(unsigned int x){ return x ? __builtin_ffs(x) : 0; }
inline int __popc(unsigned int x){ return __builtin_popcount(x); }

#define warp_serial for (int __only = 0; __only < 1; ++__only)
#else
#define warp_serial                                    \
  for (uint __mask = __ballot_sync(__activemask(), 1), \
            __num = __popc(__mask),                    \
            __lanemask = GPUTools::lanemask_lt(),      \
            __local_id = __popc(__lanemask & __mask),  \
            __active = 0;                              \
       __active < __num;                               \
       ++__active)                                     \
    if (__active == __local_id)
#endif


namespace GPUTools
{

  template<int PSIZE>
  class __PointerEquivalent
  {
  public:
    typedef unsigned int type;
  };
  template<>
  class __PointerEquivalent<8>
  {
  public:
    typedef unsigned long long int type;
  };

  typedef GPUTools::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;


  __host__ __device__ inline uint laneid()
  {
#ifdef __CUDA_ARCH__
    uint mylaneid;
    asm("mov.u32 %0, %%laneid;" : "=r" (mylaneid));
    return mylaneid;
#else
    return 0u;
#endif
  }

  __host__ __device__ inline uint warpid()
  {
#ifdef __CUDA_ARCH__
    uint mywarpid;
    asm("mov.u32 %0, %%warpid;" : "=r" (mywarpid));
    return mywarpid;
#else
    return 0u;
#endif
}
  __host__ __device__ inline uint nwarpid()
  {
#ifdef __CUDA_ARCH__
    uint mynwarpid;
    asm("mov.u32 %0, %%nwarpid;" : "=r" (mynwarpid));
    return mynwarpid;
#else
    return 1u;
#endif
  }

  __host__ __device__ inline uint smid()
  {
#ifdef __CUDA_ARCH__
    uint mysmid;
    asm("mov.u32 %0, %%smid;" : "=r" (mysmid));
    return mysmid;
#else
    return 0u;
#endif
  }

  __host__ __device__ inline uint nsmid()
  {
#ifdef __CUDA_ARCH__
    uint mynsmid;
    asm("mov.u32 %0, %%nsmid;" : "=r" (mynsmid));
    return mynsmid;
#else
    return 1u;
#endif
  }

  __host__ __device__ inline uint lanemask()
  {
#ifdef __CUDA_ARCH__
    uint lanemask;
    asm("mov.u32 %0, %%lanemask_eq;" : "=r" (lanemask));
    return lanemask;
#else
    return 0xffffffffu;
#endif
  }

  __host__ __device__ inline uint lanemask_le()
  {
#ifdef __CUDA_ARCH__
    uint lanemask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r" (lanemask));
    return lanemask;
#else
    return 0xffffffffu;
#endif
  }

  __host__ __device__ inline uint lanemask_lt()
  {
#ifdef __CUDA_ARCH__
    uint lanemask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r" (lanemask));
    return lanemask;
#else
    return 0xffffffffu;
#endif
  }

  __host__ __device__ inline uint lanemask_ge()
  {
#ifdef __CUDA_ARCH__
    uint lanemask;
    asm("mov.u32 %0, %%lanemask_ge;" : "=r" (lanemask));
    return lanemask;
#else
    return 0xffffffffu;
#endif
  }

  __host__ __device__ inline uint lanemask_gt()
  {
#ifdef __CUDA_ARCH__
    uint lanemask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r" (lanemask));
    return lanemask;
#else
    return 0xffffffffu;
#endif
  }

  template<class T>
  __host__ __device__ inline T divup(T a, T b) { return (a + b - 1)/b; }

}
