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

std::shared_ptr<HiltiImageSample> PreloadSample::load(int sampleindex, const StereoSequence* ss) const{
return std::make_shared<HiltiImageSample>(time,ss,  sampleindex, mlib::read_image1f(paths), datas);
}




std::map<float128, std::vector<imu::Data>>
collect_imu_by_frametimes(std::vector<float128> frametimes, std::vector<imu::Data> datas)
{
    // we will use all imu up to the next frametime to predict,

    std::map<float128, std::vector<imu::Data>> buckets;
    for(auto frametime:frametimes)
    {
        auto& bucket=buckets[frametime];
        bucket.reserve(1e3);
    }


    std::sort(datas.begin(), datas.end(), [](const imu::Data& a, const imu::Data& b){return a.time<b.time;});

    for(const auto& d:datas)
    {

        // iterator to the first key that is bigger or equal to the imu time.
        auto it=buckets.lower_bound(d.time);
        // imu observations preceeding the first frame are added to the first frame imu observations
        // no point in predicting the first frame, its identity anyways.
        if(it==buckets.begin()){
            it->second.push_back(d);
            continue;
        }
        // imu observations after the last frame are all added to the second to last frame
        //  a bit of extra prediction horizont wont hurt here,
        if(it==buckets.end())
        {
            auto rbegin=buckets.rbegin();
            rbegin--;
            rbegin->second.push_back(d);
            continue;
        }
        // otherwise go the the first frametime before the first frametime that is bigger or equal to the imu time, and add
        it--;
        it->second.push_back(d);
    }
    return buckets;
}



















std::map<float128, int> parse_timestamp(std::string imgpath)
{
    std::string path=imgpath+"timestamps.txt";
    if(!fs::exists(fs::path(path)))
    {
        mlog()<<"looking for: "<<path<<"\n";
        exit(1);
    }

    std::map<float128, int> mp;

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
        time*=1000*1000*1000;
        time+= ns;
        time/=1000*1000*1000;
        // followed by host time, which we discard.

        ifs>>toss;
        ifs>>toss;
        if(!ifs) break;
        auto it=mp.find(id);
        if(it!=mp.end()) {
            mlog()<<"repeated ids"<<id<<"in path: "<<path<<"\n";
        }
        //if(!fs::exists(fs::path(fpath))) {            cout<<"missing image with index: "<<id<<"on path: "<<fpath<<"\n"; continue;       }
        mp[time]=id;
    }
    return mp;
}

auto parse_alphasense_timestamps(std::string path)
{
    std::map<std::string, std::map<float128, int>> timestamps;

    timestamps["left"]  =parse_timestamp(path+"cam"+str(1)+"/image_raw/");
    timestamps["right"] =parse_timestamp(path+"cam"+str(0)+"/image_raw/");
    timestamps["cam2"]  =parse_timestamp(path+"cam"+str(2)+"/image_raw/");
    timestamps["cam3"]  =parse_timestamp(path+"cam"+str(3)+"/image_raw/");
    timestamps["cam4"]  =parse_timestamp(path+"cam"+str(4)+"/image_raw/");
    return timestamps;
}


std::vector<float128> tovec(std::map<int,float128> ts){
    std::vector<float128> t;t.reserve(ts.size());
    for(auto [id, time]:ts)
        t.push_back(time);
    return t;
}

std::string Sequence::rectified_path(std::string basepath, int camera, int index) const{
    auto it=num2path.find(camera);
    if(it==num2path.end()) {
        mlog()<<"i: "<<camera<<" "<<index<<"\n";
        exit(1);
    }
    std::string path=basepath+"/"+it->second + "/"+mlib::toZstring(index,10)+".exr";
    if(!fs::exists(path)){mlog()<<"trouble: "<<path<<"\n";exit(1);}
    return path;
}

std::map<float128, std::map<int, std::string>> Sequence::load_image_paths(std::string path) {
    // missing frames,
    std::map<std::string, std::map<float128, int>> framess=parse_alphasense_timestamps(path);
    // now we seek
    std::map<float128, std::map<int, std::string>> time2num2path;
    for(auto [name, imgs]:framess){
        for(auto [time, num]:imgs){
            int camera_num=name2num[name]; // the rectified number,
            time2num2path[time][camera_num]=path+"/"+rectified_path(path, camera_num, num);
        }
    }
    return time2num2path;
}

std::vector<imu::Data> read_imu(std::string path){
    if(!fs::exists(path)){mlog()<<"trouble: "<<path<<"\n";exit(1);}
    std::vector<imu::Data> data;
    std::ifstream ifs(path);
    while(ifs)
    {


        int imu_msg_count;
        ifs >>imu_msg_count;
        int msg_count;
        ifs >>msg_count;

        int seconds;
        ifs >>seconds;
        int nanoseconds;
        ifs >>nanoseconds;
        int seconds2;
        ifs >>seconds2;
        int nanoseconds2;
        ifs >>nanoseconds2;
        float128 time = seconds;
        time*=1000*1000*1000;
        time+= nanoseconds;
        time/=1000*1000*1000;

        Vector3d acc;
        for(int i=0;i<3;++i)            ifs>>acc[i];
        Vector3d omega;
        for(int i=0;i<3;++i)            ifs>>omega[i];
        data.push_back(imu::Data(time, acc, omega));
    }
    return data;
}



Sequence::Sequence(std::string path, std::string sequence_name): sequence_name(sequence_name)
{
    for(auto [a,b]:name2num)num2name[b]=a;
    cout<<"created hilti sequence: "<<path<<endl;
    // read metadata, create config
    cout<<"Warning, you havent read the metadata yet... "<<endl;
    // generate image paths
    std::map<float128, std::map<int, std::string>> time2num2path = load_image_paths(path);

    // read imu data.
    auto imuds=read_imu(path+"/alphasense/imu/data.txt");
    std::vector<float128> frametimes;frametimes.reserve(time2num2path.size());
    for(const auto& [time,toss]:time2num2path) frametimes.push_back(time);
    std::map<float128, std::vector<imu::Data>> frametime2imu_datas=collect_imu_by_frametimes(frametimes, imuds);
    // image time 2
    std::map<float128, PreloadSample> preload_samples_map;
    for(const auto& [time, num2path]: time2num2path)
    {
        PreloadSample& preload_sample=preload_samples_map[time];
        preload_sample.time=time;
        preload_sample.paths=num2path;
        auto it=frametime2imu_datas.find(time);
        if(it==frametime2imu_datas.end()){
            mlog()<<"humm?"<<time<<"\n";
            exit(1);
        }

        auto& ds=it->second;
        if(ds.empty()) mlog()<<"humm?\n";
        preload_sample.datas=ds;
    }
    // this automatically sorts the preload_samples too
    for(const auto& [time, pls]:preload_samples_map)
        preload_samples.push_back(pls);

    // remove the excessive timestamp offset...
    t0=preload_samples[0].time;
    for(auto& p:preload_samples) p.time-=t0;

    // generate the frameid to time mapper
    f2t=std::make_shared<Frameid2TimeMapLive>();
    for(uint i=0;i<preload_samples.size();++i)
        f2t->add(i,preload_samples[i].time);


    //alphasense done ...

    // fuck me...
    // livox  os_cloud_node  tf  tf_static  vicon
}

StereoCalibration Sequence::calibration() const{
    return StereoCalibration(rows(),cols(), fy, fx, py, px, baseline, PoseD::Identity());
}

int Sequence::samples() const{return preload_samples.size();}
int Sequence::rows() const{return 1080;}
int Sequence::cols() const{return 1440;}
std::string Sequence::name() const{return sequence_name;}

std::shared_ptr<Frameid2TimeMap> Sequence::fid2time() const
{
    return f2t;

}


std::shared_ptr<StereoSample> Sequence::sample(int index) const{

    return preload_samples.at(index).load(index, this);
}

} // end namespace hilti
} // end namespace cvl
