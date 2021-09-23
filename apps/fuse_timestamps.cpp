#include <fstream>
#include <mlib/utils/files.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/mlog/log.h>
#include <string>
#include <mlib/utils/string_helpers.h>

using std::cout;using std::endl;


std::string intstr(float128 t){
    std::stringstream ss; ss.precision(20);
    ss<<t;
    return ss.str();
}

std::map<int, float128> parse_timestamp(std::string path)
{

    if(!fs::exists(fs::path(path)))
    {
        mlog()<<"looking for: "<<path<<"\n";
        exit(1);
    }

    std::map<int, float128> mp;

    std::ifstream ifs(path);

    std::string line;
    std::getline(ifs,line);// toss header
    while(std::getline(ifs,line))
    {

        std::stringstream ss(line);

        // format is imagefilename time_in_nanoseconds_using96bits
        int imagefilename;
        int toss_int;
        float128 time_ns;
        float128 toss_time;
        ss>>imagefilename;
        ss>>toss_int;
        ss>>time_ns;
        ss>>toss_time;
        std::stringstream tss;
        tss<<imagefilename<<"\t"<<toss_int<<"\t"<<intstr(time_ns)<<"\t"<<intstr(toss_time);
        if(tss.str()!=line) {
            cout<<tss.str()<<endl;
            cout<<line<<endl;
        }
        mp[imagefilename]=time_ns;
    }
    return mp;
}


void write_metadata(std::string path){
    std::set<float128> times;
    for(int i=0;i<5;++i)
    {
        std::map<int, float128> ts = parse_timestamp(path+"cam"+str(i)+"/image_raw/timestamps.txt");
        for(auto [num,time]:ts)
            times.insert(time);
    }
    std::ofstream ofs(path+"times.txt");
    for(auto time:times)
        ofs<<intstr(time)<<"\n";
}


void rename(std::string path)
{
    for(int i=0;i<5;++i)
    {

        std::map<int, float128> ts = parse_timestamp(path+"cam"+str(i)+"/image_raw/timestamps.txt");

        std::string imdir=path+"cam"+str(i)+"/";
        for(auto [num, t]:ts)
        {
            std::string ipath=imdir+"image_rectified/"+mlib::toZstring(num,10)+".exr";

            if(fs::exists(ipath)){
                std::string opath=imdir+"image_rectified/"+intstr(t)+".exr";
                fs::rename(ipath,opath);
            }
        }
    }
}


std::vector<float128> read_times(std::string path){
    std::ifstream ifs(path);
    std::vector<float128> times;times.reserve(1e6);
    float128 time;
    while (ifs>>time) {
        times.push_back(time);
    }
    return times;
}

void rename_stereo(std::string path)
{
    {
        std::map<int, float128> ts = parse_timestamp(path+"left/timestamps.txt");
        for(auto [num, time]:ts)
        {
            std::string ipath=path+"left/left_"+mlib::toZstring(num,10)+".exr";

            if(fs::exists(ipath)){
                std::string opath=path+"left/"+intstr(time)+".exr";
                fs::rename(ipath,opath);
            }
        }
    }
    {
    std::map<int, float128> ts = parse_timestamp(path+"right/timestamps.txt");
    for(auto [num, time]:ts)
    {
        std::string ipath=path+"right/right_"+mlib::toZstring(num,10)+".exr";

        if(fs::exists(ipath)){
            std::string opath=path+"right/"+intstr(time)+".exr";
            fs::rename(ipath,opath);
        }
    }
    }
    {
    std::map<int, float128> ts = parse_timestamp(path+"left/timestamps.txt"); // because we can only guess...
    for(auto [num, time]:ts)
    {
        std::string ipath=path+"disparity/disparity_"+mlib::toZstring(num,10)+".exr";

        if(fs::exists(ipath)){
            std::string opath=path+"disparity/"+intstr(time)+".exr";
            fs::rename(ipath,opath);
        }
    }
    }
}

int main() {

    std::string path="/storage/datasets/hilti/preprocessed/Construction_Site_1/alphasense/";
    //rename(path);
    //write_metadata(path);
    //rename_stereo(path);







}
