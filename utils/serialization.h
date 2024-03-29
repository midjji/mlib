﻿#pragma once
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
#include <map>

#include <mlib/utils/checksum.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/mzip/mzip_view.h>



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
bool verify_string(const std::string& vdata);

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

template<typename T> struct bits_t { T t; }; //no constructor necessary
template<typename T> bits_t<T&> bits(T& t) {    return bits_t<T&>{t};}
template<typename T> bits_t<const T&> bits(const T& t) {    return bits_t<const T&>{t};}


template <class Value>
struct is_pair : public std::false_type{};
template <class First, class Second>
struct is_pair< std::pair<First,Second> > : public std::true_type{};

template <class Value>
struct is_map : public std::false_type{};
template <class Key, class Value,class Less, class Alloc>
struct is_map< std::map<Key,Value,Less,Alloc> > : public std::true_type{};


template<typename Stream, typename T>
Stream& operator<<(Stream& s,bits_t<T> b)
{
    if(!s) return s;
    using Type=std::remove_reference_t<T>;
    if constexpr (std::is_trivially_copyable<Type>()){
        s.write(reinterpret_cast<const char*>(&b.t), sizeof(T));
    }
    else if constexpr(is_pair<Type>()){ // add tuple one too..
        s<<bits(b.t.first);
        s<<bits(b.t.second);
    }
    else if constexpr(is_map<Type>()){ // add tuple one too..
        uint64_t size=b.t.size();
        s<<bits(size);
        for(const auto& [a,b]:b.t) {
            s<<bits(a);
            s<<bits(b);
        }
    }
    else{
        // covers vector and string and lots others...
        uint64_t size=b.t.size();
        s<<bits(size);
        for(const auto& e:b.t) s<<bits(e);
    }
    return s;
}




//extraction operator to call ::read(), require a non-const reference here
template<typename Stream, typename T>
Stream& operator>>(Stream& s, bits_t<T&> b) {
    if(!s) return s;
    using Type=std::remove_reference_t<T>;
    if constexpr (std::is_trivially_copyable<Type>()){
        s.read(reinterpret_cast<char*>(&b.t), sizeof(T));
    }
    else if constexpr(is_pair<Type>()){
        s>>bits(b.t.first);
        s>>bits(b.t.second);
    }
    else if constexpr(is_map<Type>())
    {
        b.t.clear();
        using Key= typename Type ::key_type;
        using Value= typename Type:: mapped_type;
        Key key;
        Value value;
        uint64_t size;
        s>>bits(size);
        for(uint i=0;i<size;++i) {
            s>>bits(key);
            s>>bits(value);
            b.t[key]=value;
        }
    }
    else
    {
        // so if your pod isnt trivial, why are you landing here?
        // supports string and vector, and possibly alot of other stuff that isnt actually supported by this at all... user beware!
        uint64_t size;
        s>>bits(size);

        // mlog()<<"should be a string, or a vector: "<<type_name(b.t)<<" size: "<<size<<"\n";
        b.t.resize(size);
        for(uint i=0;i<size;++i)
            s>>bits(b.t[i]);

    }
    return s;
}
template<typename T> struct vbits_t { T t; }; //no constructor necessary
template<typename T> vbits_t<T&> vbits(T& t) {    return vbits_t<T&>{t};}
template<typename T> vbits_t<const T&> vbits(const T& t) {    return vbits_t<const T&>{t};}

template<typename S, typename T>
S& operator<<(S& s,vbits_t<T> b) {
    if(!s) return s;
    // I need to have the binary as final first, there is no way around that...
    std::stringstream ss;
    ss<<bits(b.t);
    uint32_t cs=mlib::checksum32(ss.str());
    s<<bits(cs);
    s<<ss.str();
    return s;
}

template<typename S, typename T>
S& operator>>(S& s, vbits_t<T&> b) {
    if(!s) return s;
    uint32_t cs;
    s>>bits(cs);
    s>>bits(b.t);
    std::stringstream ss;
    ss<<bits(b.t);
    if(cs!=mlib::checksum32(ss.str()))
        s.setstate(std::ios::failbit);
    return s;
}




/*
#include "serialization.h"
struct inner{
    float a;
    std::uint16_t d;
}
struct pod{
    std::uint16_t a,b;
    std::uint32_t c;
    std::uint16_t d;
    float c;
    inner i;
};
void test(){
    pod p;
    ss<<bits(p);
    ss>>bits(p);
}
*/

