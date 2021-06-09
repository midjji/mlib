#pragma once
/* ********************************* FILE ************************************/
/** \file    checksum.h
 *
 * \brief    This header contains a simple checksum functionality based on murmurhash3 + unique short fix
 *
 * \remark
 * - isolated, i.e. depends on nothing but this header and its cpp
 * - fast, but not super fast for the shorter ones.
 *
 * \todo
 *
 * \example
 * std::string str;
 * std::string cs=checksum(str);
 *
 *
 *
 * \note
 * - guarantees unique hashes for short strings
 *
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2015-04-01
 * \note BSD licence
 *
 ******************************************************************************/
#include <string>
#include <array>
namespace mlib{
uint32_t checksum32(const std::string& str);
uint64_t checksum64(const std::string& str);
std::array<uint64_t, 2> checksum128(const std::string& str);
uint32_t checksum32(const uint8_t* ptr, uint len);
uint64_t checksum64(const uint8_t* ptr, uint len);


}
