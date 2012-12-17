/***
 *  $Id$
 **
 *  File: example.cpp
 *  Created: Oct 09, 2009
 *
 *  Author: Jaroslaw Zola <jaroslaw.zola@gmail.com>
 *		  Abhinav Sarje <abhinav.sarje@gmail.com>
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "io4example.hpp"
#include "sys_tools.hpp"


template <typename value_type>
void process_sym(int p, unsigned int n, unsigned int m, value_type* M, value_type* D) {
	value_type* D_ptr = D;
	value_type* x = M;
	for (unsigned int i = 0; i < n; ++i) {
		value_type* y = M;
		for (unsigned int j = 0; j < i; ++j) {
			for (unsigned int k = 0; k < m; ++k) {
				D_ptr[j] += pow(fabs(x[k] - y[k]), p);
			}
			y += m;
		} // y
		D_ptr += i;
		x += m;
	} // x
} // process_sym


template <typename value_type>
void process_nonsym(int p, unsigned int n, unsigned int m, value_type* M, value_type* D) {
	value_type* D_ptr = D;
	value_type* x = M;
	for (unsigned int i = 0; i < n; ++i) {
		value_type* y = M;
		for (unsigned int j = 0; j < n; ++j) {
			for (unsigned int k = 0; k < m; ++k) {
				D_ptr[j] += pow(fabs(x[k] - y[k]), p);
			}
			y += m;
		} // y
		D_ptr += n;
		x += m;
	} // x
} // process_nonsym


int main(int argc, char* argv[]) {
	if (argc != 7) {
	std::cout << "Usage: " << argv[0] << " n m p infile outfile sym\n";
	return 0;
	}

	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	int p = atoi(argv[3]);
	bool sym = atoi(argv[6]);

	typedef float value_type;

	unsigned int sz = n * n;
	if (sym == true) sz = ((sz - n) >> 1);

	value_type* M = new value_type[n * m];
	value_type* D = new value_type[sz];
	memset(D, 0, sz * sizeof(value_type));

	value_type pf = (value_type)1.0 / p;

	// load M from file
	if (load_M_row(argv[4], n, m, M) == false) {
	std::cerr << "Error: can't read file " << argv[4] << std::endl;
	return -1;
	}

	double T_start = jaz::get_time();

	std::cout << "* processing" << std::endl;

	if (sym == false) process_nonsym(p, n, m, M, D);
	else process_sym(p, n, m, M, D);
	for (unsigned int i = 0; i < sz; ++i) D[i] = pow(D[i], pf);

	double T_stop = jaz::get_time();
	std::cout << (T_stop - T_start) << " [s]\n";

	bool res = false;

	if (sym == false) res = store_D_nonsym(argv[5], D, n);
	else res = store_D_sym(argv[5], D, n);

	if (res == false) {
	std::cerr << "Error: can't write file " << argv[5] << std::endl;
	return -1;
	}

	delete[] D;
	delete[] M;

	return 0;
} // main
