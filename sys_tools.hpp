/***
 *  $Id: sys_tools.hpp 410 2009-10-11 04:05:23Z zola $
 **
 *  File: sys_tools.hpp
 *  Developed: Sep 17, 2005
 *
 *  Author: Jaroslaw Zola <jaroslaw.zola@gmail.com>
 *  Copyright (c) 2004-2008 Jaroslaw Zola
 *  Distributed under the Boost Software License.
 *  See accompanying file LICENSE.
 */

#ifndef JAZ_SYS_TOOLS_HPP
#define JAZ_SYS_TOOLS_HPP

#include <sstream>
#include <string>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/time.h>


namespace jaz {

  /** Simple method to check endiannes of the machine (at compilation level).
   *  @param T must be a built-in type, and sizeof(T) > 1.
   */
  template <typename T = long int> class big_endian_base {
  public:
      /** This value is true for big endian machines.
       */
      static const bool result;


  private:
      static const T one_;

  }; // class big_endian_base

  template <typename T> const T big_endian_base<T>::one_ = 1;

  template <typename T> const bool big_endian_base<T>::result
  = (!(*((char*)(&big_endian_base<T>::one_))));

  /** Interface to check endiannes.
   *  Use: if (jaz::big_endian::result == true) { // big endian code }
   */
  typedef big_endian_base<> big_endian;


  /** This function tests if std::istringstream.rdbuf()->pubsetbuf(...)
   *  works as expected.
   */
  inline bool iss_pubsetbuf_test() {
      char buf[] = "test\n";

      std::istringstream is;
      is.rdbuf()->pubsetbuf(buf, sizeof(buf));

      std::string s;
      is >> s;

      return (s.compare("test") == 0);
  } // iss_pubsetbuf_test


  /** Function to get the file size.
   *  Size of the file which does not exists is 0.
   */
  inline unsigned long int file_size(const char* name) {
      struct stat buf;
      int res = stat(name, &buf);
      return (res == 0 ? static_cast<unsigned long int>(buf.st_size) : 0);
  } // file_size


  /** Function to get size of allocated dynamic memory.
   *  @return size of allocated memory in bytes.
   */
  inline unsigned long int mem_usage() {
      struct mallinfo m = mallinfo();
      unsigned long int mem = m.uordblks + m.hblkhd;
      return mem;
  } // mem_usage


  /** Function to get the current system time.
   *  @return the number of seconds since the Epoch.
   */
  inline double get_time() {
      timeval t;
      gettimeofday(&t, 0);
      return t.tv_sec + (0.000001 * t.tv_usec);
  } // get_time

} // namespace jaz

#endif // JAZ_SYS_TOOLS_HPP
