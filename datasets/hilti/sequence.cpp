#include <thread>
#include <fstream>
#include <filesystem>

#include <mlib/utils/string_helpers.h>
#include <mlib/utils/cvl/matrix_adapter.h>

#include <mlib/datasets/hilti/sequence.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/opencv_util/read_image.h>
#include <mlib/utils/files.h>
#include <mlib/utils/vector.h>



namespace fs = std::filesystem;
using std::cout;
using std::endl;

namespace cvl{
namespace hilti{

std::shared_ptr<HiltiImageSample> PreloadSample::load(int sampleindex, const std::shared_ptr<StereoSequence>ss) const
{
    for(const auto& [id, path]:paths){
        if(!fs::exists(path))
            cout<<id<<"path: "<<path<<endl;
    }

    return std::make_shared<HiltiImageSample>(time, ss,  sampleindex, mlib::read_image1f(paths,false), datas);
}


std::tuple<float128,float128> mint(std::vector<imu::Data> datas)
{
    if(datas.empty()) return std::make_tuple<float128,float128>(0,0);
    float128 mint=datas[0].time;
    float128 maxt=datas[0].time;
    for(auto data:datas){
        if(data.time<mint) mint=data.time;
        if(data.time>maxt) maxt=data.time;
    }
    return {mint,maxt};
}

bool is_in(const std::vector<imu::Data>& datas, float128 t0, float128 t1){
    for(const auto& data:datas){
        if(data.time<t0) return false;
        if(data.time>t1) return false;
    }
    return true;
}



std::map<float128, std::vector<imu::Data>>
collect_imu_by_frametimes(
        std::vector<float128> frametimes,
        std::vector<imu::Data> datas)
{
    // we will use all imu up to the next frametime to predict,

    std::map<float128, std::vector<imu::Data>> buckets;
    for(auto frametime:frametimes)
    {
        auto& bucket=buckets[frametime];
        bucket.reserve(1e3);
    }
    std::sort(datas.begin(), datas.end(), [](const imu::Data& a, const imu::Data& b){return a.time<b.time;});

    cout.precision(20);

    cout<<"frametimes: "<<frametimes.size()<<endl;


    // all that are less or equal to the second...
    auto it=buckets.begin();
    // any before t0 are added to the first one
    float128 t0=it->first;
    for(const auto& d:datas)    {if(d.time<t0)    it->second.push_back(d);    }
    // any after t1 are added to the last one
    float128 t1=buckets.rbegin()->first;
    for(const auto& d:datas)    {if(d.time>t1)    buckets.rbegin()->second.push_back(d);    }

    while(it!=buckets.end())
    {
        float128 curr_time=it->first;
        auto nit =it; nit++;
        float128 next_time=nit->first;

        for(const auto& d:datas)
        {
            if(d.time<curr_time) continue;
            if(d.time>next_time) break;
            it->second.push_back(d);
        }
        it++;
    }





    mlog()<<"warning validate this... \n";
    for(const auto& [time, imus]:buckets)
    {
        auto tmp = mint(imus);auto minv=std::get<0>(tmp);auto maxv=std::get<1>(tmp);
        //cout<<"time: "<<time-t0<<" "<<minv-t0<<" "<<maxv-t0<<endl;
    }
    return buckets;
}




std::vector<imu::Data> read_imu(std::string path){
    if(!fs::exists(path)){mlog()<<"trouble: "<<path<<"\n";exit(1);}
    std::vector<imu::Data> data;
    std::ifstream ifs(path);
    std::string line;
    std::getline(ifs,line);
    while(ifs)
    {

        //msg_counter     msg_seq msg_header_ts   msg_recive_ts   linX    linY    linZ    angX    angY    angZ

        int imu_msg_count;
        ifs >>imu_msg_count;
        int msg_count;
        ifs >>msg_count;
        float128 time;
        ifs>>time;
        float128 toss;
        ifs>>toss;

        Vector3d acc;
        for(int i=0;i<3;++i)            ifs>>acc[i];
        Vector3d omega;
        for(int i=0;i<3;++i)            ifs>>omega[i];
        data.push_back(imu::Data(time, acc, omega));
    }
    return data;
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
std::string intstr(float128 t){
    std::stringstream ss; ss.precision(20);
    ss<<t;
    return ss.str();
}

std::vector<std::string> active_lines(std::string calib_path)
{
    std::ifstream ifs(calib_path);
    std::vector<std::string> lines;lines.reserve(10);
    std::string line;
    while(std::getline(ifs,line)){
        if(line.size()==0) continue;
        if(line[0]=='#') continue;
        lines.push_back(line);
    }
    return lines;
}

void Sequence::read_metadata(std::string path)
{
    std::string calib_path=path+"post_rectification_calibration.txt";
    cout<<"reading metadata: "<<calib_path<<endl;
    if(!fs::exists(calib_path)){
        mlog()<<"Trying to read: "<<calib_path<<"\n";
        exit(1);
    }

    std::vector<std::string> lines=active_lines(calib_path);
    std::stringstream ifs;
    for(const auto& line:lines) ifs<<" "<<line<<" ";
    cout<<ifs.str()<<endl;

    ifs>>calib.rows_;
    ifs>>calib.cols_;
    ifs>>calib.fx_;
    ifs>>calib.fy_;
    ifs>>calib.px_;
    ifs>>calib.py_;
    //mlog()<<calib.py_<<endl;
    Matrix4d m;for(int i=0;i<16;++i) ifs >>m[i];
    calib.P_left_imu_=PoseD(m);


    for(int i=0;i<16;++i)     ifs >>m[i];
    calib.P_right_imu_=PoseD(m);

    for(int i=0;i<3;++i){
        for(int j=0;j<7;++j)
            ifs>>calib.P_x_imu(i+2)[j];
    }

    mlog()<<calib.str()<<"\n";

    //mlog()<<m;
}

Sequence::Sequence(std::string path,
                   std::string sequence_name):
    sequence_name(sequence_name)
{
    for(const auto& [a,b]:name2num)num2name[b]=a;

    path=mlib::ensure_dir(path);
    path+="alphasense/";


    mlib::Timer timer("read hilti sequence timer for "+ sequence_name);
    timer.tic();

    cout<<"created hilti sequence: "<<path<<endl;
    // read metadata, create config
    read_metadata(path);


    // generate image paths

    auto times=read_times(path+"times.txt");
    std::map<float128, std::map<int, std::string>> time2num2path;
    for(auto time:times)
    {
        std::map<int, std::string> num2path;
        num2path[0] = path+"cam0/"+intstr(time)+".exr";
        num2path[1] = path+"cam1/"+intstr(time)+".exr";
        num2path[2] = path+"cam2/"+intstr(time)+".exr";
        num2path[3] = path+"cam3/"+intstr(time)+".exr";
        num2path[4] = path+"cam4/"+intstr(time)+".exr";
        num2path[5] = path+"disp/"+intstr(time)+".exr";
        time2num2path[time]=num2path;
    }



    // read imu data.
    std::map<float128, std::vector<imu::Data>> frametime2imu_datas=collect_imu_by_frametimes(times, read_imu(path+"imu/data.txt"));



    // image time 2
    std::map<float128, PreloadSample> preload_samples_map;
    for(const auto& [time, num2path]: time2num2path)
    {
        PreloadSample& preload_sample=preload_samples_map[time];
        preload_sample.time=time;
        preload_sample.paths=num2path;
        //for(auto [num, path]:num2path) cout<<"num,path: "<<num<<", "<<path<<endl;
        auto it=frametime2imu_datas.find(time);
        if(it==frametime2imu_datas.end()){
            mlog()<<"humm?"<<time<<"\n";
            exit(1);
        }

        auto& ds=it->second;
        if(ds.empty()) mlog()<<"empty image paths?\n";
        preload_sample.datas=ds;
    }
    // this automatically sorts the preload_samples too
    preload_samples.reserve(preload_samples_map.size());
    for(const auto& [time, pls]:preload_samples_map)
        preload_samples.push_back(pls);

    // remove the excessive timestamp offset...
    t0=preload_samples[0].time;
    for(auto& p:preload_samples){ p.time-=t0; p.time/=1e9;}
    for(auto& p:preload_samples) for(auto& d:p.datas) {d.time-=t0;d.time/=1e9;}

    // generate the frameid to time mapper
    f2t=std::make_shared<Frameid2TimeMapLive>();
    for(uint i=0;i<preload_samples.size();++i)
        f2t->add(i,preload_samples[i].time);

    timer.toc();
    cout<<timer<<endl;
    //alphasense done ...

    // fuck me...
    // livox  os_cloud_node  tf  tf_static  vicon
}

std::shared_ptr<Sequence> Sequence::create(std::string path, std::string sequence_name){
    auto self=std::shared_ptr<Sequence>(new Sequence(path,sequence_name));
    self->wself=self;
    return self;
}
StereoCalibration Sequence::calibration() const{
    return calib.stereo_calibration(0);
}
StereoCalibration Sequence::calibration(int index) const{
    return calib.stereo_calibration(index);
}
Calibration Sequence::hilti_calibration() const{
    return calib;
}

int Sequence::samples() const{return preload_samples.size();}
int Sequence::rows() const{return calib.rows();}
int Sequence::cols() const{return calib.cols();}
std::string Sequence::name() const{return sequence_name;}

std::shared_ptr<Frameid2TimeMap> Sequence::fid2time() const
{
    return f2t;

}


std::shared_ptr<HiltiImageSample>
Sequence::sample(int index) const
{

    return preload_samples.at(index).load(index, wself.lock());
}
std::shared_ptr<StereoSample> Sequence::stereo_sample(int index) const{
    return sample(index);
}


} // end namespace hilti
} // end namespace cvl
