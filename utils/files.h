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
#include <filesystem>
namespace fs = std::filesystem;
namespace mlib{

// file helpers, mostly boost wrappers, some will work without boost but with reduced functionality
bool fileexists(fs::path, bool verboseiffalse=true);

bool directory_exists(fs::path path, bool verboseiffalse=false);
bool pathDirectoryExists(fs::path path,bool verboseiffalse=false);
std::string getPath(fs::path path);
std::string getName(fs::path path);
void makefilepath(fs::path path);
void create_or_throw(fs::path path);
std::string ensure_dir(std::string path);


} // end namespace mlib
