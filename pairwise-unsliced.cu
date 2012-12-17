/***
 *  $Id$
 **
 *  File: example-cuda.cu
 *  Created: Apr 17, 2010
 *
 *  Author: Abhinav Sarje <abhinav.sarje@gmail.com>
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iomanip>

#include "io4example.hpp"

// kernel functions for the cuda device
template<typename value_type>
__device__ __host__
value_type lpnorm(const value_type* a, const value_type* b, unsigned int d, int p, value_type inv_p) {
	value_type temp = 0.0;
	for(unsigned int k = 0; k < d; ++ k) {
		temp += pow(fabs(a[k] - b[k]), p);
	} // for

	return pow(temp, inv_p);
} // lpnorm()


template<typename value_type>
__global__
void process_nonsym(int p, unsigned int n, unsigned int d, 
		value_type* M, value_type inv_p, value_type* D, int c, int r) {
	// a block is c X c, and is responsible to compute a tile of size r X c
	// (following the weird y X x convention)

	// number of subtiles
	// for now assume that r is multiple of c to avoid boundry conditions
	//unsigned int num_subtiles = ceilf((float)r / c);
	unsigned int num_subtiles = r / c;

	unsigned int a_idx = c * blockIdx.x + threadIdx.x;
	unsigned int b_idx = r * blockIdx.y + threadIdx.y; // r is tile dimension in y direction

	value_type *a, *b;

	if(a_idx < n && b_idx < n) {
		// first get the x vector (these will remain same for the whole tile)
		a = M + d * a_idx; // the x vector
	} // if

	// shared memory for y vectors
	extern __shared__ value_type shared_M[];

	value_type* M_ptr = M + (r * d * blockIdx.y);
	unsigned int thread_pos = c * threadIdx.y + threadIdx.x;
	unsigned int shared_b_idx = d * threadIdx.y;

	// for all subtiles - this brings in the sequential part to reuse x vectors
	for(unsigned int subtile = 0; subtile < num_subtiles; ++ subtile) {
		// fill the shared memory with the subtile y vectors
		unsigned int num_transfers = ceilf((float)d / c); // ((c*d)/(c*c)), hence, will be best when d is a multiple of c

		__syncthreads();

		for(unsigned int i = 0; i < num_transfers; ++ i) {
			unsigned int index = i * c * c + thread_pos;
			if(index < c * d) shared_M[index] = M_ptr[index];
		} // for

		__syncthreads();

		if(a_idx < n && b_idx < n) {
			// now get the y vector from shared_M
			b = shared_M + shared_b_idx;

			// perform computation
			unsigned int D_idx = n * (b_idx + c * subtile) + a_idx;
			if(D_idx < n * n) D[D_idx] = lpnorm(a, b, d, p, inv_p);
		} // if

		M_ptr += c * d;
	} // for

} // process_nonsym()


// kernel for cuda device
template<typename value_type>
__global__
void process_sym(int p, unsigned int n, unsigned int d, value_type* M, value_type inv_p, value_type *D) {
} // process_sym()


// on host
template <typename value_type>
__host__
void process_nonsym_host(int p, unsigned int n, unsigned int d, value_type* M, value_type inv_p, value_type* D) {

    value_type* D_ptr = D;
    value_type* x = M;
    for (unsigned int i = 0; i < n; ++i) {
        value_type* y = M;
        for (unsigned int j = 0; j < n; ++j) {
            for (unsigned int k = 0; k < d; ++k) {
                D_ptr[j] += pow(fabs(x[k] - y[k]), p);
            }
			D_ptr[j] = pow(D_ptr[j], inv_p);
            y += d;
        } // y
        D_ptr += n;
        x += d;
    } // x
} // process_nonsym


/**
 * Miscellaneous stuff
 */

void device_information() {
	int device_count;
	cudaGetDeviceCount(&device_count);

	for(int i = 0; i < device_count; ++ i) {
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, i);

		if(i == 0) {
			if(device_prop.major == 9999 && device_prop.minor == 9999)
				std::cout << "There is no device supporting CUDA." << std::endl;
			else if(device_count == 1)
				std::cout << "There is 1 device supporting CUDA" << std::endl;
			else std::cout << "There are " << device_count << " devices supporting CUDA" << std::endl;
		}
	} // for
} // device_information()


/**
  * Main stuff
  */

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " n d p infile outfile sym\n";
        return 0;
    } // if

    int n = atoi(argv[1]);
    int d = atoi(argv[2]);
    int p = atoi(argv[3]);
    bool sym = atoi(argv[6]);

    typedef float value_type;

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << "Free: " << (float)free/(1024*1024*1024) << ", Total: " << (float)total/(1024*1024*1024) << std::endl;

	//device_information();

    unsigned int out_size = n * n;
    if (sym == true) out_size = ((out_size - n) >> 1);

	// use page-locked host memory
	value_type* M;
	if(cudaHostAlloc((void **) &M, n * d * sizeof(value_type), 0) != cudaSuccess) {
		std::cerr << "Error: cannot allocate page-locked memory." << std::endl;
		return -1;
	} // if
	value_type* D;
	if(cudaHostAlloc((void **) &D, out_size * sizeof(value_type), cudaHostAllocWriteCombined) != cudaSuccess) {
		std::cerr << "Error: cannot allocate page-locked memory." << std::endl;
		return -1;
	} // if
	// set output to 0
	if(memset(D, 0, out_size * sizeof(value_type)) == NULL) {
		std::cerr << "Error: memset failed." << std::endl;
		return -1;
	} // if

	// allocate device memories
	value_type* d_M;
	value_type* d_D;
	if(cudaMalloc((void **) &d_M, n * d * sizeof(value_type)) != cudaSuccess) {
		std::cerr << "Error: cannot allocate device memory." << std::endl;
		return -1;
	} // if
	if(cudaMalloc((void **) &d_D, out_size * sizeof(value_type)) != cudaSuccess) {
		std::cerr << "Error: cannot allocate device memory." << std::endl;
		return -1;
	} // if

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime = 0.0;

    value_type inv_p = (value_type)1.0 / p;

    // load M from file
    if (load_M_row(argv[4], n, d, M) == false) {
        std::cerr << "Error: can't read file " << argv[4] << "." << std::endl;
        return -1;
    } // if

	cudaEventRecord(start, 0);

	// copy M from host to device
	// and set device output memory to 0
	cudaMemcpy(d_M, M, n * d * sizeof(value_type), cudaMemcpyHostToDevice);
	cudaMemset(d_D, 0, out_size * sizeof(value_type));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "Device Memory Time: " << elapsedTime << "ms" << std::endl;

    std::cout << "* processing ..." << std::endl;

//	for(int i = 32; i <= 512; i+=32) {
//		std::cout << std::setw(3) << i << ": ";
//		for(int j = 1; j <= i; j *= 2) {

			cudaEventRecord(start, 0);

			// set grid and block sizes
			//dim3 block(i/j, j);
			const int BLOCK_SIZE = 16;
			const int TILE_MUL = 4;

			int c = BLOCK_SIZE;
			int r = c * TILE_MUL;	// doesnt really have to be multiple of r

			dim3 block(c, c);
			dim3 grid(ceil((float)n / block.y), ceil((float)n / r));
			size_t s_M_size = c * d * sizeof(value_type);

			std::cout << "Block: (" << block.y << ", " << block.x << ")" << std::endl;
			std::cout << "Grid: (" << grid.y << ", " << grid.x << ")" << std::endl;
			std::cout << "num_subtiles: " << ceilf((float)r / c) << std::endl;

			// invoke the kernel function
    		std::cout << "* on device" << std::endl;
	    	if(sym == false) process_nonsym<<< grid, block, s_M_size >>>(p, n, d, d_M, inv_p, d_D, c, r);
	    	else process_sym<<< grid, block >>>(p, n, d, d_M, inv_p, d_D);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsedTime, start, stop);
			std::cout << "Device Compute Time: " << elapsedTime << "ms" << std::endl;
//			std::cout << "[" << std::setw(3) << block.x << "," << std::setw(3) << block.y << "] ";
//			std::cout << std::fixed << std::setprecision(3) << elapsedTime << " ms. ";
//		}
//		std::cout << std::endl;
//	}

	cudaEventRecord(start, 0);

	// trasnfer results (D) from device memory to host memory
	cudaMemcpy(D, d_D, out_size * sizeof(value_type), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "Device Memory Time: " << elapsedTime << "ms" << std::endl;

	/*cudaEventRecord(start, 0);
    std::cout << "* on host" << std::endl;
	process_nonsym_host(p, n, d, M, inv_p, D);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "Host Time: " << elapsedTime << "ms" << std::endl;
	*/

    bool res = false;
    if (sym == false) res = store_D_nonsym(argv[5], D, n);
    else res = store_D_sym(argv[5], D, n);

    if (res == false) {
        std::cerr << "Error: can't write file " << argv[5] << std::endl;
        return -1;
    } // if

	// free the events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// free host and device memories
	cudaFree(d_D);
	cudaFree(d_M);
	cudaFreeHost(D);
	cudaFreeHost(M);

    return 0;
} // main
