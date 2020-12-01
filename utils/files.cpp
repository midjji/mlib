#include "mlib/utils/files.h"


#include <fstream>

#include <iostream>

namespace fs = std::experimental::filesystem;
using std::cout;using std::endl;



namespace mlib{


/* //4x faster filecheck on linux
bool file_exists(const std::string& name){
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}
*/
bool fileexists(fs::path path, bool verboseiffalse){

    if(fs::exists(path)) return true;
    if(verboseiffalse)
        std::cout<<"\nFile not found: "<<path<<endl;
    return false;
}

bool directory_exists(fs::path path, bool verboseiffalse){

    if (fs::is_directory(path)) return true;
    if(fs::exists(path.parent_path()))return true;
    if(verboseiffalse)
        std::cout<<"\nDirectory not found: "<<path<<std::endl;
    return false;
}
std::string getPath(std::experimental::filesystem::path path){
    return path.parent_path();
}
std::string getName(std::experimental::filesystem::path path){
    return path.stem()/path.extension();
}

void makefilepath(fs::path path){
    fs::create_directories(path);
}
/**
 * @brief isImageExtension
 * @param ext
 * @return
 *
 * this is by .xxx not metadata, for common variants of png && jpg
 */
bool isImageExtension(fs::path path){
    std::string ext=path.extension();
    if(ext==".png")
        return true;
    if(ext==".jpg")
        return true;
    if(ext==".jpeg")
        return true;
    if(ext==".JPG")
        return true;
    if(ext==".JPEG")
        return true;
    if(ext==".exr") return true;
    if(ext==".pgm") return true;

    return false;
}


}//end namespace mlib

