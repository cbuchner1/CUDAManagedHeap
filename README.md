# Cuda Managed Heap

A heap for CUDA using managed memory space that works for CPU and GPU objects alike.


## About

This is a fork of the ScatterAlloc memory heap that was modernized to also support SM 7.0 and later.
It was modified to live in managed memory (the heap structure itself as well as the blocks it allocates).

This also makes the heap usable from the host, allowing entire objects to reside in managed memory so
they can be freely shared with the GPU, crossing host-device boundaries at will. For this to work, your
objects need to have members decorated with \_\_host\_\_ \_\_device\_\_ and need to provide custom new and delete
allocators that place the objects themselves on the managed heap.

The alloc() and dealloc() members can also be called by host code. System wide atomics are used
in CUDA code to allow the CPU and GPU access the heap concurrently. The host side uses operations
from the \<atomic\> header.


## Possible use cases

* For passing data containers between CPU and GPU in a producer / consumer pattern.

* When unable to determine in advance how much memory would be required to return a result, allocate a container on the GPU and return it with exactly the size required.

* For implementing e.g. STL-like storage containers that work on CPU and GPU alike.

* For applications that need to do a lot of dynamic memory allocations on GPU. ScatterAlloc has very high performance as it is mostly contention free. See the related paper under Resources.

* For applications that like to share heap space between CPU and GPU


## Additional Notes

Note that the unit tests I added are very bare-bones, as I do not have access to the original test and
benchmarking suite done by the authors at TU Graz.

Also note that my main test environment is using clang++-20 as the CUDA compiler. The CMakeLists.txt may
require some modifications to work with other compilers


## Possible Future Improvemenets

* Dynamically growing the heap as needed (before it runs out of space) would be a nice thing to implement.

* Adding more rigorous testing will be required to validate for correctness and torture test the concurrency capabilities.


## Resources

An archived copy of the research paper behind ScatterAlloc is found
[here.](https://web.archive.org/web/20160201114513/http://www.icg.tugraz.at/Members/steinber/scatteralloc-1)

The MVP project web page has been archived
[here.](https://web.archive.org/web/20170311124644/http://www.icg.tugraz.at/project/mvp/)
