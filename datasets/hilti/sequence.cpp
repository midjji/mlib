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
#include <vector>



namespace fs = std::filesystem;
using std::cout;
using std::endl;

namespace cvl{
namespace hilti{

PreloadSample::PreloadSample(int sampleindex, float128 time,
                             float128 original_time_ns,
               std::vector<imu::Data> datas,
                             std::map<int,std::string> cam2paths ):
    sampleindex(sampleindex),time_(time),original_time_ns_(original_time_ns),datas(datas),cam2paths(cam2paths)
{

}


bool PreloadSample::has(int index) const{
    auto it=cam2paths.find(index);
    require(it!=cam2paths.end(),"wtf are you doing... ");
    return fs::exists(it->second);
}
bool PreloadSample::has_all() const{
    for(int i=0;i<6;++i){
        if(!has(i)) return false;
    }
    return true;
}
std::string PreloadSample::str() const
{
    int sampleindex=0;//?
    std::stringstream ss;
    ss<<"sample: "<<sampleindex<<" of time: "<<time()<<" and "<<original_time_ns();
    if(has_all())
        ss<<" has all images.\n";
    else{
        ss<<"\n";
        for(const auto& [id, path]:cam2paths)
        {
            if(!fs::exists(path))
                ss<<"is missing: "<<id<< " with path: "<<path<<"\n";
        }
    }
    // should be sorted, but lets not assume...
    auto data_copy=datas;

    if(datas.empty())
        ss<< "sample: "<<sampleindex<<" is missing imu data\n";
    else{
        std::sort(data_copy.begin(), data_copy.end(),[](imu::Data a, imu::Data b){return a.time<b.time;});
        ss<<" and has imu data: "<<datas.size()<<" spanning: "<< data_copy[0].time<<" to "<<data_copy.rbegin()[0].time<<"\n";
        ss<<"verify its been sorted!\n";
    }
    return ss.str();
}
std::shared_ptr<HiltiImageSample> PreloadSample::load(const std::shared_ptr<StereoSequence> ss) const {
    if(!has_all()) cout<<str()<<endl;



    auto images=mlib::read_image1f(cam2paths,false);
    // multiply the regular images by 16
    for(int i=0;i<5;++i)
    {
        auto it=images.find(i);
        if(it==images.end()) continue;
        it->second=it->second*16.0f;
    }

    return std::make_shared<HiltiImageSample>(time(), ss,  sampleindex, images, datas, original_time_ns());
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
        if(line.empty()) continue;
        if(line[0]=='#') continue;
        lines.push_back(line);
    }
    return lines;
}
std::string numbers(std::vector<std::string> lines){
    std::stringstream ss;
    for(const auto& line:lines)
        ss<<" "<<line<<" ";
    std::string str;

    for(char c:ss.str()) if(c!=',')str.push_back(c);
    return str;
}



void Sequence::read_metadata(std::string path)
{
    std::string calib_path=path+"post_rectification_calibration.txt";
    cout<<"reading metadata: "<<calib_path<<endl;
    if(!fs::exists(calib_path)){
        mlog()<<"Trying to read: "<<calib_path<<"\n";
        exit(1);
    }

    std::stringstream ss(numbers(active_lines(calib_path)));
    //cout<<ss.str()<<endl;

    ss>>calib.rows_;
    ss>>calib.cols_;
    ss>>calib.fx_;
    ss>>calib.fy_;
    ss>>calib.px_;
    ss>>calib.py_;
    //mlog()<<calib.py_<<endl;
    Matrix4d m;for(int i=0;i<16;++i) ss >>m[i];
    calib.P_left_imu_=PoseD(m);


    for(int i=0;i<16;++i)     ss >>m[i];
    calib.P_right_imu_=PoseD(m);
    PoseD P_right_left=calib.P_right_imu_*calib.P_left_imu_.inverse();
    calib.baseline_=-P_right_left.t()[0];


    for(int i=0;i<3;++i){
        for(int j=0;j<7;++j)
            ss>>calib.P_x_imu(i+2)[j];
    }
    // they are child2parent

    for(int i=0;i<3;++i)
        calib.P_x_imu(i+2).invert();

    for(int i=0;i<5;++i){
        double len=calib.P_x_imu(i).q().length();
        if(std::abs(len-1)>1e-5) {
            cout<<"q length: "<<calib.P_x_imu(i).q().length()<<endl;
            exit(1);
        }
    }

    //mlog()<<"\nRead Hilti Calibration: \n"<<calib.str()<<"\n";

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
    // sort them? I should not need to!
    float128 t0=times[0];

    std::map<float128, std::map<int, std::string>> time2num2path;
    for(auto time:times)
    {
        std::map<int, std::string> num2path;
        num2path[0] = path+"left/"+intstr(time)+".exr";
        num2path[1] = path+"right/"+intstr(time)+".exr";
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
    int sampleindex=0;
    for(const auto& [time, num2path]: time2num2path)
    {
        auto it=frametime2imu_datas.find(time);
        if(it==frametime2imu_datas.end()){
            mlog()<<"humm?"<<time<<"\n";
            exit(1);
        }
        auto& ds=it->second;
        if(ds.empty()) mlog()<<"image without imu data"<<time<<"\n";

        preload_samples_map[time] = PreloadSample(sampleindex++,(time-t0)/1e9,time,ds,num2path);
    }
    // this automatically sorts the preload_samples too
    preload_samples.reserve(preload_samples_map.size());
    for(const auto& [time, pls]:preload_samples_map)
        preload_samples.push_back(pls);


    // generate the frameid to time mapper
    f2t=std::make_shared<Frameid2TimeMapLive>();
    for(uint i=0;i<preload_samples.size();++i)
        f2t->add(i,preload_samples[i].time());

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

StereoCalibration Sequence::calibration() const
{
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
    return preload_samples.at(index).load(wself.lock());
}
std::shared_ptr<StereoSample> Sequence::stereo_sample(int index) const{
    return sample(index);
}


} // end namespace hilti
} // end namespace cvl
