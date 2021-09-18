#include <thread>
#include <fstream>
#include <filesystem>

#include <mlib/utils/string_helpers.h>
#include <mlib/utils/cvl/matrix_adapter.h>

#include <mlib/datasets/hilti/sequence.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/opencv_util/read_image.h>



namespace fs = std::filesystem;
using std::cout;
using std::endl;

namespace cvl{
namespace hilti{



std::map<float128, std::string> parse_timestamp(std::string imgpath)
{
    std::string path=imgpath+"timestamps.txt";
    if(!fs::exists(fs::path(path)))
    {
        mlog()<<"looking for: "<<path<<"\n";
        exit(1);
    }

    std::map<float128, std::string> mp;

    std::ifstream ifs(path);
    while(ifs)
    {
        // format is int  int  uint64_t uint64_t
        //           toss toss seconds  nanoseconds

        int64_t id=-1;
        ifs>> id;
        int64_t toss;
        ifs >> toss;
        std::uint64_t s,ns;
        ifs>>s;
        ifs>>ns;
        float128 time = s;
        time*=1000000000;
        time+= ns;
        time/=1000000000;
        // followed by host time, which we discard.

        ifs>>toss;
        ifs>>toss;
        if(!ifs) break;
        auto it=mp.find(id);
        if(it!=mp.end()) {
            mlog()<<"repeated ids"<<id<<"in path: "<<path<<"\n";
        }
        std::string fpath=imgpath+mlib::toZstring(id,10)+".png";
        if(!fs::exists(fs::path(fpath))) {
            cout<<"missing image with index: "<<id<<"on path: "<<fpath<<"\n"; continue;
        }
        mp[time]=fpath;
    }
    return mp;
}

auto parse_alphasense_timestamps(std::string path)
{
    std::vector<std::map<float128, std::string>> timestamps;
    for(int i=0;i<5;++i) {
        auto cpath=path+"cam"+str(i)+"/image_raw/";
        timestamps.push_back(parse_timestamp(cpath));
    }
    return timestamps;
}


std::vector<float128> tovec(std::map<int,float128> ts){
    std::vector<float128> t;t.reserve(ts.size());
    for(auto [id, time]:ts)
        t.push_back(time);
    return t;
}

auto load_image_paths(std::string path)
{    
    // there are framedrops everywhere,
    // there are partial framedrops everywhere
    // there may be missing images
    // pita to use, so lets only use the ones that are available.
    std::set<float128> times;
    std::vector<std::map<float128,std::string>> framess=parse_alphasense_timestamps(path);
    for(const auto& frames:framess)
        for(const auto& frame:frames)
            times.insert(frame.first);

    std::map<float128,std::map<int, std::string>> common;
    for(float128 time:times)
    {
        std::map<int,std::string> imgs;
        for(int i=0;i<5;++i){
            auto it=framess[i].find(time);
            if(it==framess[i].end()) continue;
            imgs[i]=it->second;
        }
        common[time]=imgs;
    }
    return common;
}

void Sequence::prepare_image_paths(){
    if(path.size()> 0 && path.back()!='/')
        path.push_back('/');
    image_paths =  load_image_paths(path+"alphasense/");
    int i=0;
    for(const auto& [time,toss]:image_paths){
        index2time[i++]=time;
    }
    cout<<"loaded: "<<index2time.size()<<"index2time and "<<image_paths.size()<<endl;
}


Sequence::Sequence(std::string path):path(path){
    cout<<"created hilti sequence: "<<path<<endl;
    prepare_image_paths();

    // read metadata
    //alphasense  livox  os_cloud_node  tf  tf_static  vicon
}

int Sequence::samples() const{return image_paths.size();}

std::shared_ptr<Sample> Sequence::sample(int index) const{
    auto it=index2time.find(index);
    if(it==index2time.end()){
        mlog()<<"bad index!"<<index<<"\n";
        exit(1);
    }
    float128 time=it->second;
    auto it2=image_paths.find(time);
    if(it2==image_paths.end()){
        mlog()<<"missing time for known index!\n";
        exit(1);
    }


    return std::make_shared<ImageSample>(time,mlib::read_image1b(it2->second));
}

} // end namespace hilti
} // end namespace cvl
