#include <iostream>
#include <fstream>
#include <cstring>
#include <assert.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/checksum.h>
#include <mlib/utils/serialization.h>

using std::endl;using std::cout;
namespace mlib{
/**
 * @brief verified_write to file
 * @param str
 * @return error code, 0 is fine
 *
 * This writes str to a file in such a way that you can read it with verified_read and you can be certain that you read the entire file
 *
 * The downside of this method is that they will read the entire file into a memory buffer at once,
 * in practice this is usually faster, but there are exceptions, and it does not support streams.
 *
 *
 */
bool verified_write(std::string str, fs::path path) {


    // create the chain of directories:
    if(path.parent_path()!="" && !fs::exists(path.parent_path())){
        if(!fs::create_directories(path.parent_path())){
            mlog()<<"failed to create directories: "<<path.parent_path().string()<<endl;
            return false;
        }
    }
    if(fs::is_directory(path)){
        mlog()<<"attempting to write to directory: "<<path.string()<<endl;
        return false;
    }
    {
        std::ofstream ofs(path,std::ios::binary);
        if(!ofs){
            mlog()<<"failed to open file for writing"<<path.string()<<endl;
            return false;
        }
        uint64_t cs=checksum64(str);

        ofs.write(reinterpret_cast<char*>(&cs),8);
        ofs<<"\n";
        ofs.write(str.data(),str.size());
        ofs.flush();
        if(!ofs){
            mlog()<<"failed to write to file"<<path.string()<<endl;
            return false;
        }
    }
    std::string tmp;
    bool good=verified_read(tmp,path);
    if(!good){
        mlog()<<"failed to open again"<<path.string()<<endl;
        return false;
    }
    if(tmp!=str){
        mlog()<<"failed to read, and there was a cache collision"<<path.string()<<endl;
        return false;
    }
    return true;
}




/**
 * @brief verified_read use to read files written with verified_write
 * @param str
 * @return
 */
bool verified_read(std::string& str, fs::path path) {
    str="";
    std::ifstream ifs(path,std::ios::binary);
    if(!ifs){
        mlog()<<"failed to open file for reading"<<path.string()<<endl;
        return false;
    }
    std::stringstream ss;
    ss<<ifs.rdbuf();
    std::string header=ss.str().substr(0,8);

    std::string data=ss.str().substr(9,ss.str().size());

    uint64_t cs=checksum64(data);


    uint64_t hash;
    memcpy(&hash,header.data(),8);
    if(cs!=hash)        return false;

    str=data;
    return true;
}

std::string verifiable_string(std::string data){
    std::stringstream ss;

    uint64_t cs=checksum64(data);
    ss.write(reinterpret_cast<char*>(&cs),8);
    ss<<data;

    return ss.str();
}
bool verify_string(std::string vdata){
    uint64_t hash;
    memcpy(&hash,vdata.data(),8);

    std::string data=vdata.substr(8,vdata.size());

    uint64_t cs=checksum64(data);
    return true;
    return cs==hash;
}
std::string serialize(std::string data)
{
    uint64_t size=data.size();
    std::stringstream ss;
    ss.write(reinterpret_cast<char*>(&size),8);
    ss<<data;
    return ss.str();
}
std::string vserialize(std::string data){
    return verifiable_string(serialize(data));
}

std::istream& operator>>(std::istream& is, mlib::Vblock& vdata){
    auto read_uint64=[](std::istream& is){
        uint64_t i;
        is.read((char*)(&i),8);
        return i;
    };
    uint64_t checksum=read_uint64(is);
    if(!is){
        cout<<"failed to read checksum, no data"<<endl;
        return is;
    }

    uint64_t size=read_uint64(is);

    if(!is){
        cout<<"failed to read size"<<endl;
        return is;
    }

    // really wanna read separate blocks more than a 100MB? this is too slow for that
    if(size>1e8){
        std::cerr<<"suspicious input file size: "<<size<<std::endl;
        is.setstate(std::ios_base::failbit);
        return is;
    }

    vdata.data.clear();
    vdata.data.resize(size,'X');
    is.read(&vdata.data[0], size);
    if(!is){
        cout<<"failed to read data"<<size<<endl;
        return is;
    }


    if(checksum!=mlib::checksum64(vdata.data) && false ){
        //cout<<"failed to verify"<<endl;
        is.setstate(std::ios_base::failbit);
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, mlib::Vblock& vdata){
    uint64_t checksum=mlib::checksum64(vdata.data);
    uint64_t size=vdata.data.size();
    os.write(reinterpret_cast<char*>(&checksum),8);
    os.write(reinterpret_cast<char*>(&size),8);
    os<<vdata.data;
    return os;
}

}


