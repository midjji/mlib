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
#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <sstream>




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
bool verified_write(std::string str, std::experimental::filesystem::path path);
/**
 * @brief verified_read use to read files written with verified_write
 * @param str
 * @return
 */
bool verified_read(std::string& str,std::experimental::filesystem::path path);


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

// write basic types i.e. int, float etc, but not all pods! as bits to iostream
//struct to hold the value:
template<typename T> struct bits_t { T t; }; //no constructor necessary
//functions to infer type, construct bits_t with a member initialization list
//use a reference to avoid copying. The non-const version lets us extract too
template<typename T> bits_t<T&> bits(T &t) { return bits_t<T&>{t}; }
template<typename T> bits_t<const T&> bits(const T& t) { return bits_t<const T&>{t}; }
// now for basic types thats it!, note wont even work for all pods... see pragma pack
// I should probably restrict it, but its so nice so its up to the user!

//insertion operator to call ::write() on whatever type of stream
template<typename S, typename T>
S& operator<<(S& s,bits_t<T> b) {
    s.write(reinterpret_cast<char*>(&b.t), sizeof(T));
    return s;
}
//extraction operator to call ::read(), require a non-const reference here
template<typename S, typename T>
S& operator>>(S& s, bits_t<T&> b) {
    s.read(reinterpret_cast<char*>(&b.t), sizeof(T));
    return s;
}

