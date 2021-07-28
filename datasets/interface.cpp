#include <mlib/datasets/interface.h>
#include <mlib/utils/mlog/log.h>
namespace cvl {


Frameid2TimeMap::~Frameid2TimeMap(){}
std::string str(const Frameid2TimeMap& f2t){return f2t.str();}



double Frameid2TimeMapLive::time_of(int fid) const
{

    auto it=fid2time.find(fid);
    if(it!=fid2time.end()) return it->second;
    mlog()<<"frame id: "<<fid<< " not found, where you trying to predict?";
    return predict_time_of(fid);
}
int Frameid2TimeMapLive::frameid_of(double time_seconds) const
{
    auto it=time2fid.find(time_seconds);
    if(it!=time2fid.end()) return it->second;

    mlog()<<"time :"<<time_seconds<<" not found, where you trying to predict?\n";
    return predict_frameid_of(time_seconds);
}

double Frameid2TimeMapLive::predict_time_of(int frameid) const
{
    auto it=fid2time.find(frameid);
    if(it!=fid2time.end()) return it->second;
    return frameid*10.0;// fix later, this is for kitti only...
}

int Frameid2TimeMapLive::predict_frameid_of(double time) const {
    auto it=time2fid.find(time);
    if(it!=time2fid.end()) return it->second;

    return time/10.0;// fix later, this is for kitti only...
}


void Frameid2TimeMapLive::add(int fid, double ts)
{
    //mlog()<<"\n"<<fid<< " "<<ts<<"\n";
    auto fit=fid2time.find(fid);
    if(fit!=fid2time.end())
        mlog()<<"fid collision\n";
    auto tit=time2fid.find(ts);
    if(tit!=time2fid.end())
        mlog()<<"time collision\n";
    fid2time[fid]=ts;
    time2fid[ts] = fid;
}
std::string Frameid2TimeMapLive::str() const{
    std::stringstream ss;
    for(const auto& [fid, time] : fid2time)
        ss<<"fid: ("<<fid<<", "<<time<<")\n";
    for(const auto& [time,fid] : time2fid)
        ss<<"time: ("<<time<<", "<<fid<<")\n";
    return ss.str();
}


}
