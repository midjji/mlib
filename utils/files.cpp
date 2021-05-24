#include <fstream>
#include <iostream>

#include "mlib/utils/files.h"
#include <mlib/utils/mlog/log.h>
using std::cout;using std::endl;
namespace mlib{


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
std::string getPath(fs::path path){
    return path.parent_path();
}
std::string getName(fs::path path){
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

void create_or_throw(fs::path path){        
    if(path.filename()!="" )
        path=path.remove_filename();
    try {
        fs::create_directories(path);
    }  catch (fs::filesystem_error& fe) {
    mlog()<<"failed to create path: \""<<path.string()<< "\" with error "<<fe.what()<<endl;
    }
}

std::string ensure_dir(std::string path){
    if(path.size()==0) return path;
    if(path.back()!='/')
        path.push_back('/');
    return path;
}
}//end namespace mlib

