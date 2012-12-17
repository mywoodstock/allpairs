/***
 *  File: pairwise-general.cu
 *  Created: Apr 17, 2010
 *
 *  Author: Abhinav Sarje <abhinav.sarje@gmail.com>
 *
 *  Notes: Simple implementation. No sliding, subtiling.
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iomanip>

#include "io4example.hpp"

// kernel functions for the cuda device
template<typename value_type>
__device__
value_type lpnorm_slice(const value_type* a, const value_type* b, unsigned int d, int p) {
	value_type temp = 0.0;
	for(unsigned int k = 0; k < d; ++ k) {
		temp += pow(fabs(a[k] - b[k]), p);
	} // for

	return temp;
} // lpnorm_slice()

template<typename value_type>
__device__
value_type lpnorm_acc(const value_type v, value_type inv_p) {
	return pow(v, inv_p);
} // lpnorm_acc()


template<typename value_type>
__global__
void process_nonsym(int p, unsigned int n, unsigned int d, 
		value_type* M, value_type inv_p, value_type* D, int M_pitch = 0, int D_pitch = 0) {

	unsigned int a_idx = (blockDim.x * blockIdx.x + threadIdx.x);
	unsigned int b_idx = (blockDim.y * blockIdx.y + threadIdx.y);
#ifndef PITCHED
	unsigned int Ma_idx = d * a_idx;
	unsigned int Mb_idx = d * b_idx;
	unsigned int D_idx = n * a_idx + b_idx;
	unsigned int max_idx = d * n;

	if((Ma_idx < max_idx) && (Mb_idx < max_idx)) {
		// get the two vectors and the output index
		value_type* a = M + Ma_idx;
		value_type* b = M + Mb_idx;

		D[D_idx] = lpnorm_slice(a, b, d, p);
		D[D_idx] = lpnorm_acc(D[D_idx], inv_p);
	} // if
#else
	// for pitched memory - is not being useful at all
	unsigned int Ma_idx = M_pitch * a_idx;
	unsigned int Mb_idx = M_pitch * b_idx;
	unsigned int D_idx = D_pitch * a_idx + b_idx;
	unsigned int max_idx = M_pitch * n;

	if((Ma_idx < max_idx) && (Mb_idx < max_idx)) {
		// get the two vectors and the output index
		value_type* a = M + Ma_idx;
		value_type* b = M + Mb_idx;

		D[D_idx] = lpnorm_slice(a, b, d, p);
		D[D_idx] = lpnorm_acc(D[D_idx], inv_p);
	} // if
#endif
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

	device_information();

    unsigned int out_size = n * n;
    if (sym == true) out_size = ((out_size - n) >> 1);

	// allocate host memories
    /*value_type* M = new value_type[n * d];
    value_type* D = new value_type[out_size];
    memset(D, 0, out_size * sizeof(value_type));*/

	// try using page-locaked host memory
	value_type* M;
	cudaHostAlloc((void **) &M, n * d * sizeof(value_type), 0);
	value_type* D;
	cudaHostAlloc((void **) &D, out_size * sizeof(value_type), cudaHostAllocWriteCombined);
	memset(D, 0, out_size * sizeof(value_type));

	// allocate device memories
	value_type* d_M;
	value_type* d_D;
#ifndef PITCHED
	cudaMalloc((void **) &d_M, n * d * sizeof(value_type));
	cudaMalloc((void **) &d_D, out_size * sizeof(value_type));
#else
	// try using pitched memory
	size_t M_pitch, D_pitch;
	cudaMallocPitch((void **) &d_M, &M_pitch, d * sizeof(value_type), n);
	cudaMallocPitch((void **) &d_D, &D_pitch, n * sizeof(value_type), n);

	std::cout << "d bytes: " << d * sizeof(value_type) << ", M_pitch: " << M_pitch << std::endl;
	std::cout << "n bytes: " << n * sizeof(value_type) << ", D_pitch: " << D_pitch << std::endl;
#endif

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime = 0.0;

    value_type inv_p = (value_type)1.0 / p;

    // load M from file
    if (load_M_row(argv[4], n, d, M) == false) {
        std::cerr << "Error: can't read file " << argv[4] << std::endl;
        return -1;
    } // if

	cudaEventRecord(start, 0);

	// copy M from host to device
	// and set device output memory to 0
#ifndef PITCHED
	cudaMemcpy(d_M, M, n * d * sizeof(value_type), cudaMemcpyHostToDevice);
	cudaMemset(d_D, 0, out_size * sizeof(value_type));
#else
	cudaMemcpy2D(d_M, M_pitch, M, d * sizeof(value_type), d * sizeof(value_type), n, cudaMemcpyHostToDevice);
	cudaMemset2D(d_D, D_pitch, 0, n * sizeof(value_type), n);
#endif

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
			dim3 block(8, 8);
			dim3 grid(ceil((float)n / block.x), ceil((float)n / block.y));

			// invoke the kernel function
    		std::cout << "* on device" << std::endl;
#ifndef PITCHED
	    	if (sym == false) process_nonsym<<< grid, block >>>(p, n, d, d_M, inv_p, d_D);
	    	else process_sym<<< grid, block >>>(p, n, d, d_M, inv_p, d_D);
#else
	    	if (sym == false) process_nonsym<<< grid, block >>>(p, n, d, d_M, inv_p, d_D, M_pitch / sizeof(value_type), D_pitch / sizeof(value_type));
	    	else process_sym<<< grid, block >>>(p, n, d, d_M, inv_p, d_D);
#endif

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
#ifndef PITCHED
	cudaMemcpy(D, d_D, out_size * sizeof(value_type), cudaMemcpyDeviceToHost);
#else
	cudaMemcpy2D(D, n * sizeof(value_type), d_D, D_pitch, n * sizeof(value_type), n, cudaMemcpyDeviceToHost);
#endif

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
    /*delete[] D;
    delete[] M;*/
	cudaFreeHost(D);
	cudaFreeHost(M);

    return 0;
} // main
