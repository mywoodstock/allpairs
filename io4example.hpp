/***
 *  $Id: io4example.hpp 411 2009-10-11 18:28:15Z zola $
 **
 *  File: io4example.hpp
 *  Created: Oct 11, 2009
 *
 *  Author: Jaroslaw Zola <jaroslaw.zola@gmail.com>
 *          Abhinav Sarje <abhinav.sarje@gmail.com>
 */

#ifndef IO4EXAMPLE_HPP
#define IO4EXAMPLE_HPP

#include <fstream>

template <typename Float>
bool load_M_row(const char* name, int n, int m, Float* M) {
    std::cout << "* reading " << name << std::endl;

    std::ifstream f(name);
    if (!f) return false;

    Float* row = M;
    unsigned int row_sz = m;
    for (int i = 0; i < n; ++i, row += row_sz) {
        for (int j = 0; j < m; ++j) f >> row[j];
    }

    f.close();
    return true;
} // load_M_row


/* template <typename Float>
bool load_M_row128(const char* name, int n, int m, Float* M) {
    std::cout << "* reading " << name << std::endl;

    std::ifstream f(name);
    if (!f) return false;

    Float* row = M;
    unsigned int row_sz = pnorm_size128(m, sizeof(Float));
    for (int i = 0; i < n; ++i, row += row_sz) {
        for (int j = 0; j < m; ++j) f >> row[j];
    }

    f.close();
    return true;
} */ // load_M_row128


template <typename Float>
bool store_D_sym(const char* name, const Float* D, int n) {
    std::cout << "* writing " << name << std::endl;

    std::ofstream f(name);
    if (!f) return false;

    Float* tab = const_cast<Float*>(D);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j, ++tab) f << (*tab) << "\t";
        f << std::endl;
    }

    f.close();
    return true;
} // store_D_nonsym


template <typename Float>
bool store_D_nonsym(const char* name, const Float* D, int n) {
    std::cout << "* writing " << name << std::endl;

    std::ofstream f(name);
    if (!f) return false;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) f << D[i * n + j] << "\t";
        f << std::endl;
    }

    f.close();
    return true;
} // store_D_nonsym

#endif // IO4EXAMPLE_HPP
