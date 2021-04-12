#pragma once
/* ********************************* FILE ************************************/
/** \file    files.hpp
 *
 * \brief    This header contains compat functions for the old filesystem wrappers
 *
 * \remark
 * - c++11
 *
 * \todo
 * - update old code accordingly
 *
 *
 *
 * \author   Mikael Persson,
 * \date     2015-04-01
 *
 ******************************************************************************/
#include <string>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
namespace mlib{

// file helpers, mostly boost wrappers, some will work without boost but with reduced functionality
bool fileexists(std::experimental::filesystem::path, bool verboseiffalse=true);

bool directory_exists(std::experimental::filesystem::path path, bool verboseiffalse=false);
bool pathDirectoryExists(std::experimental::filesystem::path path,bool verboseiffalse=false);
std::string getPath(std::experimental::filesystem::path path);
std::string getName(std::experimental::filesystem::path path);
void makefilepath(std::experimental::filesystem::path path);
void create_or_throw(std::experimental::filesystem::path path);


} // end namespace mlib
