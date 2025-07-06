# Cuda Managed Heap

A heap for CUDA using managed memory space that works for CPU and GPU objects alike.


## About

This is a fork of the ScatterAlloc memory heap that was modernized to also support SM 7.0 and later.
It was modified to live in managed memory (the heap structure itself as well as the blocks it allocates).

This also makes the heap usable from the host, e.g. allowing to place entire objects in managed memory by
so they can be freely shared with the GPU, crossing host device boundaries at will. For this to work your
objects need to have members decorated with \_\_host\_\_ \_\_device\_\_ and need to provide custom new and delete
allocators that place the objects themselves on the managed heap.

## Additional Notes

Note that the unit testing I added is very barebones, as I do not have access to the original test and
benchmarking suite done by the authors at TU Graz.

Also note that my main test environment is using clang++-20 as the CUDA compiler. The CMakeLists.txt may
require some modifications to work with other compilers

## Possible Future Improvements

The host specific alloc and dealloc functions currently call into device kernels instead of providing a
performant, CPU-optimized implementation of these functions.

An code example showing passing C++ objects across device boundaries would be nice to have.

## Resources

An archived copy of the research paper behind ScatterAlloc is found
[here.](https://web.archive.org/web/20160201114513/http://www.icg.tugraz.at/Members/steinber/scatteralloc-1)

The MVP project web page has been archived
[here.](https://web.archive.org/web/20170311124644/http://www.icg.tugraz.at/project/mvp/)
