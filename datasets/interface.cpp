#include <mlib/datasets/interface.h>
#include <mlib/utils/mlog/log.h>
namespace cvl {


Frameid2TimeMap::~Frameid2TimeMap(){}




double Frameid2TimeMapLive::time_of(int fid) const
{
    auto it=fid2time.find(fid);
    if(it==fid2time.end()) {
        mlog()<<"frame id not found, where you trying to predict?\n";
        return 0;
    }
    return it->second;
}
double Frameid2TimeMapLive::predict_time_of(int frameid) const
{
    auto it=fid2time.find(frameid);
    if(it!=fid2time.end()) { // perfect prediction then...
        return it->second;        
    }
    return 0;
}
int Frameid2TimeMapLive::frameid_of(double time_seconds) const{return 0;}
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
int Frameid2TimeMapLive::predict_frameid_of(double time) const {return 0;}

}
