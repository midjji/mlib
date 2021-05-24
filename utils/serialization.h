#pragma once
/* ********************************* FILE ************************************/
/** \file    serialization.h
 *
 * \brief    This header contains a simple verified write/read file functions.
 *
 * \remark
 * - c++11
 * - stdc++fs
 * - uses mlib/utils/checksum
 *
 * \todo
 * - convert to c++14
 *
 * \example
 * if(verified_write("some data", filepath);
 * \note
 * - verified means a hash check std::hash, i.e. not guaranteed to be non colliding and not recovereable on failure.
 *
 *
 *
 * \author   Mikael Persson
 * \date     2015-04-01
 * \note BSD licence
 *
 ******************************************************************************/
#include <string>

#include <iostream>
#include <fstream>
#include <sstream>
#include <mlib/utils/files.h>



namespace mlib{
/**
 * @brief verified_write to file
 * @param str
 * @return error code, 0 is fine
 *
 * This writes str to a file in such a way that you can read it with verified_read
 *  and you can be certain that you read the entire file
 *
 * it will also create any directories as needed
 *
 * The downside of this method is that they will read the entire file into a memory buffer at once,
 * in practice this is usually faster, but there are exceptions, and it does not support streams.
 *
 * These functions can throw, but will return false for most common failures.
 *
 *
 */
bool verified_write(std::string str, fs::path path);
/**
 * @brief verified_read use to read files written with verified_write
 * @param str
 * @return
 */
bool verified_read(std::string& str,fs::path path);


std::string verifiable_string(std::string data);
bool verify_string(std::string vdata);

std::string serialize(std::string data);

// should not be used as a replacement for untrusted datavalidation, but it is really convenient for that
std::string vserialize(std::string data);

struct Vblock{   
    std::string data;
};
// in the same namespace as the second argument, ths
std::istream& operator>>(std::istream& is, mlib::Vblock& vdata);
std::ostream& operator<<(std::ostream& os, mlib::Vblock& vdata);
}

